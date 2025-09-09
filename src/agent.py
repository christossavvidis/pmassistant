import logging
import requests
import aiohttp
import asyncio

from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a weather information assistant.
            Your ONLY purpose is to ask for a location and relay the EXACT temperature data that you receive from the weather API lookup tool.
            Do NOT add any information beyond what the API returns.
            Do NOT use any general knowledge about weather or locations.
            Simply greet, ask for location, and report the exact temperature from the API response.""",
        )

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Use this tool to look up current weather information in the given location.

        Args:
            location: The location to look up weather information for (e.g. city name)
        """
        try:
            logger.info(f"Starting weather lookup for location: {location}")
            url = f"https://wttr.in/{location}?format=%t"

            logger.info(f"Making API request to: {url}")

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    logger.info(f"API response status code: {response.status}")

                    if response.status == 200:
                        temperature = (await response.text()).strip()

                        if not temperature:
                            logger.error("Empty temperature response received")
                            return "I'm sorry, I received an empty response from the weather service."

                        logger.info(f"Successfully retrieved temperature: {temperature} for {location}")
                        return f"The current temperature in {location} is {temperature}."

                    elif response.status == 404:
                        logger.error(f"Location not found: {location}")
                        return f"I'm sorry, I couldn't find the location: {location}"

                    elif response.status == 429:
                        logger.error("Rate limit exceeded for weather API")
                        return "I'm sorry, we've exceeded the weather service rate limit. Please try again later."

                    else:
                        error_msg = f"Failed to get weather data, status code: {response.status}"
                        logger.error(error_msg)
                        raise Exception(error_msg)

        except asyncio.TimeoutError:
            error_msg = f"Timeout while getting weather for {location}"
            logger.error(error_msg)
            return "I'm sorry, the weather service is taking too long to respond. Please try again."

        except aiohttp.ClientError as e:
            error_msg = f"Connection error while getting weather for {location}: {str(e)}"
            logger.error(error_msg)
            return "I'm sorry, I couldn't connect to the weather service. It might be down."

        except Exception as e:
            error_msg = f"Unexpected error getting temperature for {location}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"I'm sorry, I couldn't get the temperature for {location} due to an unexpected error."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all providers at https://docs.livekit.io/agents/integrations/llm/
        llm=openai.LLM(model="gpt-4o-mini"),
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all providers at https://docs.livekit.io/agents/integrations/stt/
        stt=deepgram.STT(model="nova-3", language="multi"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all providers at https://docs.livekit.io/agents/integrations/tts/
        tts=cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532"),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead:
    # session = AgentSession(
    #     # See all providers at https://docs.livekit.io/agents/integrations/realtime/
    #     llm=openai.realtime.RealtimeModel()
    # )

    # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
    # when it's detected, you may resume the agent's speech
    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/integrations/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/integrations/avatar/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))