import asyncio
import io

import cv2
import discord
from discord.ext import commands, tasks

from utils import load_toml_as_dict, save_dict_as_toml
from brawl_industry_logger import get_logger, pop_logs

log = get_logger("brawl_industry.discord")


def get_panel_text(bot_state) -> str:
    status = "Running" if bot_state.is_running() else "Stopped"
    return (
        f"Brawl Industry Control Panel\n"
        f"**Status:** {status}\n"
    )


class ControlPanelView(discord.ui.View):

    def __init__(self, bot_state):
        super().__init__(timeout=None)
        self.bot_state = bot_state

    @discord.ui.button(label="Start", style=discord.ButtonStyle.green, custom_id="bi:start")
    async def start_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.bot_state.is_running():
            await interaction.response.send_message("Bot is already running.", ephemeral=True)
            return
        self.bot_state.set_running(True)
        log.info("Bot started by user")
        await interaction.response.edit_message(content=get_panel_text(self.bot_state), view=self)
        await interaction.followup.send("Bot started.", ephemeral=True)

    @discord.ui.button(label="Stop", style=discord.ButtonStyle.red, custom_id="bi:stop")
    async def stop_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self.bot_state.is_running():
            await interaction.response.send_message("Bot is already stopped.", ephemeral=True)
            return
        self.bot_state.set_running(False)
        log.info("Bot stopped by user")
        await interaction.response.edit_message(content=get_panel_text(self.bot_state), view=self)
        await interaction.followup.send("Bot stopped.", ephemeral=True)

    @discord.ui.button(label="Restart game", style=discord.ButtonStyle.gray, custom_id="bi:restart")
    async def restart_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self.bot_state.restart_func:
            await interaction.response.send_message("Restart not available yet.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.bot_state.restart_func)
            log.info("Brawl Stars restarted by user")
            await interaction.followup.send("Brawl Stars restarted.", ephemeral=True)
        except Exception as e:
            log.error(f"Restart failed: {e}")
            await interaction.followup.send("Restart failed. Check logs.", ephemeral=True)

    @discord.ui.button(label="Screenshot", style=discord.ButtonStyle.blurple, custom_id="bi:screenshot")
    async def screenshot_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not getattr(self.bot_state, "get_screenshot", None):
            await interaction.response.send_message("Screenshot engine not ready.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        try:
            result = self.bot_state.get_screenshot()
            frame  = result[0] if isinstance(result, tuple) else result
            if frame is None:
                await interaction.followup.send("Frame is empty.", ephemeral=True)
                return
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode(".png", bgr)
            if not success:
                await interaction.followup.send("Encode failed.", ephemeral=True)
                return
            file = discord.File(io.BytesIO(buffer), filename="screenshot.png")
            await interaction.followup.send("Screenshot:", file=file, ephemeral=True)
        except Exception as e:
            log.error(f"Screenshot failed: {e}")
            await interaction.followup.send("Screenshot failed. Check bot logs.", ephemeral=True)


def create_discord_bot(bot_state):
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

    bot._notify_channel  = None
    bot._log_channel     = None
    bot._view_registered = False

    @bot.event
    async def on_ready():
        log.info(f"Logged in as {bot.user}")

        if not bot._view_registered:
            bot.add_view(ControlPanelView(bot_state))
            bot._view_registered = True

        if not check_stuck_alert.is_running():
            check_stuck_alert.start()
        if not check_notifications.is_running():
            check_notifications.start()
        if not drain_logs.is_running():
            drain_logs.start()

        saved_id = load_toml_as_dict("cfg/general_config.toml").get("log_channel_id")
        if saved_id:
            for guild in bot.guilds:
                ch = guild.get_channel(int(saved_id))
                if ch:
                    bot._log_channel = ch
                    break

    @bot.event
    async def on_disconnect():
        log.warning("Discord connection lost, reconnecting")

    async def _send_alert(message: str):
        if bot._notify_channel:
            try:
                await bot._notify_channel.send(message)
                return
            except Exception:
                bot._notify_channel = None

    @bot.command(name="panel")
    @commands.cooldown(1, 3, commands.BucketType.user)
    async def panel_command(ctx: commands.Context):
        bot._notify_channel = ctx.channel
        view = ControlPanelView(bot_state)
        await ctx.send(get_panel_text(bot_state), view=view)

    @bot.command(name="bi_help")
    @commands.cooldown(1, 3, commands.BucketType.user)
    async def help_command(ctx: commands.Context):
        await ctx.send(
            "```\n"
            "Brawl Industry Commands\n"
            "-----------------------\n"
            "!panel      control panel (start / stop / restart / screenshot)\n"
            "!stats      session stats\n"
            "!resetstats reset wins/losses\n"
            "!setlogs    stream console logs in this channel\n"
            "!bi_help    this message\n"
            "```"
        )

    @bot.command(name="resetstats")
    @commands.cooldown(1, 3, commands.BucketType.user)
    async def reset_command(ctx: commands.Context):
        if bot_state.reset_stats_func:
            bot_state.reset_stats_func()
            await ctx.send("Stats reset to 0.")
        else:
            await ctx.send("Reset not available right now.")

    @bot.command(name="stats")
    @commands.cooldown(1, 3, commands.BucketType.user)
    async def stats_command(ctx: commands.Context):
        status      = "running" if bot_state.is_running() else "stopped"
        ips         = f"{bot_state.get_ips():.2f}" if bot_state.get_ips() else "..."
        stats       = bot_state.session_stats_func() if bot_state.session_stats_func else {}
        total_secs  = stats.get("time_played", 0.0)
        h, rem      = divmod(int(total_secs), 3600)
        m, s        = divmod(rem, 60)
        session_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

        await ctx.send(
            "```\n"
            "Session Stats\n"
            "-------------\n"
            f"status    : {status}\n"
            f"brawler   : {bot_state.get_current_brawler() or '...'}\n"
            f"ips       : {ips}\n"
            f"session   : {session_str}\n"
            f"wins      : {stats.get('wins', 0)}\n"
            f"losses    : {stats.get('losses', 0)}\n"
            f"draws     : {stats.get('draws', 0)}\n"
            f"winrate   : {stats.get('winrate', 0.0):.1f}%\n"
            f"games     : {stats.get('games', 0)}\n"
            "```"
        )

    @bot.command(name="setlogs")
    @commands.cooldown(1, 3, commands.BucketType.user)
    async def setlogs_command(ctx: commands.Context):
        bot._log_channel = ctx.channel
        cfg = load_toml_as_dict("cfg/general_config.toml")
        cfg["log_channel_id"] = ctx.channel.id
        save_dict_as_toml(cfg, "cfg/general_config.toml")
        await ctx.send(f"Logs redirected to {ctx.channel.mention}", delete_after=5)

    @tasks.loop(seconds=2)
    async def drain_logs():
        ch = getattr(bot, "_log_channel", None)
        if not ch:
            return
        lines = pop_logs(max_lines=20)
        if not lines:
            return
        body  = "\n".join(lines)
        limit = 1900
        while body:
            chunk, body = body[:limit], body[limit:]
            try:
                await ch.send(f"```\n{chunk}\n```")
            except Exception:
                break

    @tasks.loop(seconds=3)
    async def check_stuck_alert():
        if not bot_state.take_stuck_alert():
            return
        discord_id = load_toml_as_dict("cfg/general_config.toml").get("discord_id", "").strip()
        ping = f"<@{discord_id}> " if discord_id.isdigit() else ""
        await _send_alert(
            f"{ping}**[Brawl Industry]** Bot stuck, Brawl Stars could not restart. "
            "Press **Start** to resume after fixing your emulator."
        )

    @tasks.loop(seconds=10)
    async def check_notifications():
        msg = bot_state.pop_notification()
        while msg is not None:
            await _send_alert(f"**[Brawl Industry]** {msg}")
            msg = bot_state.pop_notification()

    @bot.event
    async def on_command_error(ctx, error):
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(f"wait {error.retry_after:.0f}s", delete_after=3)
        elif isinstance(error, commands.CheckFailure):
            pass
        else:
            raise error

    return bot
