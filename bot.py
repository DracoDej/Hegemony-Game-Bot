import discord
from discord import app_commands, Interaction, Embed, File
from discord.ext import commands
from discord.ext.commands.context import Context
from discord import Reaction, Attachment
from discord.threads import Thread
from discord.abc import GuildChannel, PrivateChannel
from discord.ui import View, Button
import gspread
import gspread.utils
from gspread.worksheet import Worksheet
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timezone, timedelta

from typing import Literal, Union, List, get_args, Dict, Tuple, Optional, Any, Callable, Set
from collections import Counter
import sys, os
import asyncio
from math import ceil
from dotenv import load_dotenv
import asyncio
from gspread.exceptions import APIError
import random
import re

# -------------- CONFIGURATION SECTION --------------
load_dotenv()  # Load variables from .env
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
SHEET_KEY = os.getenv("SHEET_KEY")

MAIN_SHEET = "Main"
BUILDINGS_SHEET = "Buildings"

ALLOWED_ROLES = ["Mod", "Helper", "Admin"]  # Roles allowed to edit silver

# Resources
FOOD = "Food"
FUEL = "Fuel"
GEMS = "Gems"
METAL = "Metal"
STONE = "Stone"
TIMBER = "Timber"
SPAWNS_USED = "Spawns Used"

# Military
BRIGADE_CAP = "Brigade Cap"
GENERAL_CAP = "General Cap"
SHIP_CAP = "Ship Cap"

# Nation
DISCORD_NAME = "Discord Name"
NATION_NAME = "Nation Name"
REGION = "Region"
RELIGION = "Religion"
CULTURE = "Culture"
TITLE = "Title"
CAPITAL = "Capital"
CITIES = "Cities"
T1_CITY_COUNT = "T1_City_Count"
T2_CITY_COUNT = "T2_City_Count"
T3_CITY_COUNT = "T3_City_Count"
TILE_COUNT = "Tile Count"
SILVER = "Silver"

resources = Literal[
    FOOD,
    FUEL,
    GEMS,
    METAL,
    STONE,
    TIMBER,
    SPAWNS_USED
]

military = Literal[
    BRIGADE_CAP,
    GENERAL_CAP,
    SHIP_CAP
]

nation = Literal[
    DISCORD_NAME,
    NATION_NAME,
    REGION,
    RELIGION,
    CULTURE,
    TITLE,
    CAPITAL,
    CITIES,
    T1_CITY_COUNT,
    T2_CITY_COUNT,
    T3_CITY_COUNT,
    TILE_COUNT,
    SILVER
]

unit_type = Union[nation, resources, military]

RESOURCE_COLUMNS = {
    FOOD: 11,
    FUEL: 12,
    GEMS: 13,
    METAL: 14,
    STONE: 15,
    TIMBER: 16
}

SHEET_COLUMNS = {
    # -------------- VIEW SHEET --------------
    DISCORD_NAME: 1,
    NATION_NAME: 2,
    REGION: 3,
    RELIGION: 4,
    CULTURE: 5,
    TITLE: 6,
    CAPITAL: 7,
    CITIES: 8,
    TILE_COUNT: 9,
    SILVER: 10,
    FOOD: 11,
    FUEL: 12,
    GEMS: 13,
    METAL: 14,
    STONE: 15,
    TIMBER: 16,
    SPAWNS_USED: 17,
    # -------------- BUILDING SHEET --------------
    T1_CITY_COUNT: 3,
    T2_CITY_COUNT: 4,
    T3_CITY_COUNT: 5,
    BRIGADE_CAP: 6,
    GENERAL_CAP: 7,
    SHIP_CAP: 8
}

normalized_unit_map = {
    key.strip().replace(" ", "").lower(): key for key in SHEET_COLUMNS.keys()
}

EXPANSION_CHANNEL_ID = "expansion_channel_id"
BUILDING_CHANNEL_ID = "building_channel_id"
RESOURCE_CHANNEL_ID = "resource_channel_id"

config = {
    EXPANSION_CHANNEL_ID: None,  # Store the ID of the expansion log channel
    BUILDING_CHANNEL_ID: None,  # Store the ID of the building log channel
    RESOURCE_CHANNEL_ID: None   # Store the ID of the resource spawn log channel
}

NUMBER_OF_SPAWNS = 3

# -------------- GOOGLE SHEETS SETUP --------------
# Define the scope
scope = ["https://www.googleapis.com/auth/spreadsheets"]

creds = ServiceAccountCredentials.from_json_keyfile_name(os.path.join(os.getcwd(), "credentials.json"), scope)
client = gspread.authorize(creds)


main_sheet = client.open_by_key(SHEET_KEY).worksheet(MAIN_SHEET)
buildings_sheet = client.open_by_key(SHEET_KEY).worksheet(BUILDINGS_SHEET)

# -------------- DISCORD BOT SETUP --------------
class UncutHelpCommand(commands.HelpCommand):
    """
    A custom help command that doesn't truncate docstrings or arguments.
    """

    def get_ending_note(self):
        """
        Remove or override the built-in ending note if you like.
        """
        return "Use !help <command> for more info on a command."

    async def send_bot_help(self, mapping):
        """
        Sends a list of all commands when the user just types !help.
        """
        # Example: build a string or multiple messages from docstrings
        help_text = []
        for cog, commands_ in mapping.items():
            filtered = await self.filter_commands(commands_, sort=True)
            command_signatures = [self.get_command_signature(c) for c in filtered]
            if command_signatures:
                cog_name = getattr(cog, "qualified_name", "No Category")
                help_text.append(f"**{cog_name}**")
                help_text.extend(command_signatures)
                help_text.append("")  # blank line

        # Join them and send as a single message
        full_help = "\n".join(help_text)
        await self.get_destination().send(full_help)

    async def send_command_help(self, command):
        """
        Sends help information for a specific command (e.g. !help status).
        """
        # Use command.help, command.signature, etc. to build your own text
        help_text = (
            f"**Command:** {command.qualified_name}\n"
            f"**Usage:** {self.get_command_signature(command)}\n\n"
            f"{command.help or 'No details provided.'}"
        )
        await self.get_destination().send(help_text)

    async def send_group_help(self, group):
        """
        Sends help for a group command.
        """
        # Similar to send_command_help, but can loop over subcommands
        subcommands = group.commands
        help_text = (
            f"**Group:** {group.qualified_name}\n\n"
            f"{group.help}\n\n"
            "Subcommands:\n"
        )
        for subcommand in subcommands:
            help_text += f"  - {subcommand.name}: {subcommand.short_doc}\n"

        await self.get_destination().send(help_text)

    async def send_cog_help(self, cog):
        """
        Sends help for all commands in a specific Cog.
        """
        commands_ = cog.get_commands()
        help_text = [f"**Cog:** {cog.qualified_name}\n", cog.description, ""]
        for command in commands_:
            help_text.append(f"{self.get_command_signature(command)} - {command.short_doc}")
        full_text = "\n".join(help_text)
        await self.get_destination().send(full_text)


intents = discord.Intents.default()
intents.members = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)
#bot.help_command = UncutHelpCommand()
tree = bot.tree


# Helper Functions

async def send(
    ctx: Context,
    msg: str | None = None,
    embed: Embed | None = None,
    file: File | None = None,
    files: list[File] | None = None,
    ephemeral: bool = False,
    mention_author: bool = False,
):
    return  await ctx.send(
            content=msg, embed=embed, file=file, files=files, mention_author=mention_author
        )
    if isinstance(ctx.interaction, Interaction):
        print("interaction")
        if not ctx.interaction.response.is_done():
            print("not done")
            await ctx.interaction.response.send_message(
                content=msg, embed=embed, file=file, files=files, ephemeral=ephemeral
            )
        else:
            print("fallback")
            # Fallback for follow-up messages if the response is already sent
            await ctx.interaction.followup.send(content=msg, embed=embed, file=file, files=files, ephemeral=ephemeral)
    else:
        print("context")
        await ctx.send(
            content=msg, embed=embed, file=file, files=files, mention_author=mention_author
        )

def author(ctx: commands.Context):
    """
    Retrieves the author/user who invoked the command, regardless of the context type.

    Args:
        ctx: The command context, which can be either a Context or an Interaction.

    Returns:
        The invoking user as a discord.Member or discord.User object.
    """
    return ctx.author

def get_unit(string: str) -> unit_type | None :
    """Returns the key corresponding to the normalized string or None if invalid."""
    search_normalized = string.strip().replace(" ", "").lower()
    return normalized_unit_map.get(search_normalized)

def get_sheet(unit: unit_type) -> Worksheet:
    """
    Returns the appropriate sheet given a unit type
    """
    if unit in get_args(nation): return main_sheet
    if unit in get_args(resources): return main_sheet
    if unit in get_args(military): return buildings_sheet
    else: return None

def batch_reset_column(sheet: gspread.Worksheet, column_name: str):
    # 1) Get the column index from your SHEET_COLUMNS mapping
    col_index = SHEET_COLUMNS[column_name]

    # 2) Determine the total rows
    
    num_rows = len(sheet.get_all_values())  # includes header
    if num_rows < 2:
        return  # No data rows

    # 3) Read only that column range (excluding headers). 
    #    For example, from row 2 to row N. If header is row 1, data starts row 2.
    cell_range = f"{gspread.utils.rowcol_to_a1(2, col_index)}:{gspread.utils.rowcol_to_a1(num_rows, col_index)}"

    # 4) Modify in Python
    # Replace each existing value with "0"
    #for i in range(len(column_data)):
     #   column_data[i] = [0]  # one-element list with 0
    column_data = [[0]]*(num_rows - 1)

    # 5) Write back in batch
    sheet.update(column_data, cell_range)

def get_user_balance(user: str, unit: unit_type) -> str | None :
    """
    Returns a given's user's unit balance.
    """

    sheet = get_sheet(unit)

    try:
        # Get all values in the column for user IDs
        users = sheet.col_values(SHEET_COLUMNS[DISCORD_NAME])  
        user_row = users.index(user) + 1  # Convert index to 1-based row number
    except:
        return None

    record = sheet.cell(user_row, SHEET_COLUMNS[unit]).value


    return record

def set_user_balance(user: str, value: str, unit: unit_type) -> str | None :
    """
    Sets a given's user's unit balance.
    """
    sheet = get_sheet(unit)
    
    try:
        # Get all values in the column for user IDs
        user_ids = sheet.col_values(SHEET_COLUMNS[USER_ID])  
        user_row = user_ids.index(str(user_id)) + 1  # Convert index to 1-based row number
    except:
        return None

    # Check if value is numeric
    try:
        value = int(value)
    except ValueError:
        value = value

    sheet.update_cell(user_row, SHEET_COLUMNS[unit], value)
    return value

def add_user_balance(user: str, amount: str, unit: unit_type) -> str | None :
    """
    Adds some amount of a unit to a given's user's unit balance.
    """
    current = get_user_balance(user, unit)
    if current is None:
        return None

    try:
        # Convert both current and amount to integers for addition
        current = int(current)
        amount = int(amount)
        sum = current + amount
    except ValueError:
        # If either value is non-numeric, treat as concatenated strings
        sum = f"{current}, {amount}"

    return set_user_balance(user, sum, unit)
"""
def log_transaction(user: str, change_amount: str, unit: unit_type, source: str, editor_id: int, editor_name: str, result: str, message_link: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    row = [
        user,         # Discord name
        timestamp,         # Date
        change_amount,     # Amount
        unit,              # Type
        source,            # Source
        str(editor_id),    # EditorID
        editor_name,       # Editor Discord name
        result,            # Result
        message_link       # Message Link
    ]

    log_sheet.append_row(row)


    return result
"""
def is_authorized(ctx: Context) -> bool:
    """
    Check if the user invoking the command has one of the allowed roles or administrator permissions.

    Args:
        ctx: The context of the command, which can be either a Context or an Interaction.

    Returns:
        bool: True if the user is authorized, False otherwise.
    """
    invoking_user = author(ctx)  # Unified way to get the invoking user

    # Check if the user has administrator permissions
    if hasattr(invoking_user, "guild_permissions") and invoking_user.guild_permissions.administrator:
        return True

    # Check if the user has one of the allowed roles
    if hasattr(invoking_user, "roles"):
        for role in invoking_user.roles:
            if role.name in ALLOWED_ROLES:
                return True

    return False

def show_log(entry: Dict[str, int | float | str], all_params: bool):
    if all_params:
        line = (f"- **User:** {entry['Discord name']} | **Change:** {entry['Amount']} {entry['Type']} | "
                f"**Source:** {entry['Source']} | **Editor:** {entry['Editor Discord name']} | "
                f"**Time:** {entry['Date']} | **New Balance:** {entry['Result']} {entry['Type']}")
    else:
        line = (f"- **User:** {entry['Discord name']} | **Change:** {entry['Amount']} {entry['Type']} | "
                f"**Source:** {entry['Source']} | "
                f"**Time:** {entry['Date']}")
    return line

def group_costs(costs: List[Tuple[int, unit_type]]) -> List[Tuple[int, unit_type]]:
    """
    Groups costs by the same unit type, summing quantities for each unit type.

    Args:
        costs: A list of (quantity, unit_type) tuples.

    Returns:
        A list of grouped (quantity, unit_type) tuples, where each unit_type appears at most once.
    """
    grouped = Counter()

    for (quantity, unit) in costs:
        grouped[unit] += quantity  # Sum quantities for the same unit type
   
    # Convert the dictionary back to a list of tuples
    return [(quantity, unit) for unit, quantity in grouped.items()]

def parse_resource_list(resource_string: str) -> Tuple[List[Tuple[int, unit_type]], List[str]]:
    """
    Given a string like: "100 Silver, 2 Crops"
    returns a list of (resource_name, amount), e.g. [(100, "Silver"), (2, "Crops")].

    - Splits by commas for multiple resources.
    - Each part should look like "<qty> <resource name>".
      e.g. "100 Silver" or "2 Crops".
    - If parsing fails, quantity defaults to 0, and the resource name is taken from what remains.
    """
    resource_string = resource_string.strip()
    if not resource_string:
        return [], []
    
    invalid_resources = []
    resource_list = []

     # Split by commas and process each part
    for part in resource_string.split(","):
        tokens = part.strip().split(None, 1)  # Split into quantity and resource name

        if len(tokens) != 2:  # Ensure we have both quantity and resource
            invalid_resources.append(part.strip())
            continue

        # Parse the quantity
        try:
            qty = int(tokens[0])
        except ValueError:
            qty = 0  # Default quantity to 0 if parsing fails

        # Normalize and validate the resource name
        resource_name = tokens[1].strip()
        unit = get_unit(resource_name)  # Convert to standardized unit
        if unit is None:
            invalid_resources.append(resource_name)
            continue

        # Add to the resource list
        resource_list.append((qty, unit))

    return resource_list, invalid_resources

async def check_debt(ctx: Context, member: discord.Member, costs: List[Tuple[int, unit_type]], auto_cancel: bool = False, extra_message: str = "", title="**Cost Summary**\n"):
    
    # group costs by sheet
    # e.g. sheet_costs = { sheet_obj: { unit: qty, ... }, ... }
    sheet_costs: Dict[gspread.Worksheet, Dict[unit_type, int]] = {}
    for (qty, unit) in costs:
        sheet_obj = get_sheet(unit)
        if not sheet_obj:
            continue
        if sheet_obj not in sheet_costs:
            sheet_costs[sheet_obj] = {}
        sheet_costs[sheet_obj][unit] = sheet_costs[sheet_obj].get(unit, 0) + qty

    user_id_str = str(member.id)
    cost_summary_lines = []
    insufficient: List[unit_type] = []


    try:
        user_ids = next(iter(sheet_costs)).col_values(SHEET_COLUMNS[USER_ID])
        # find row
        row_index = user_ids.index(user_id_str) + 1
    except ValueError:
        await send(ctx, f"User {member.display_name} does not exist. Use !create to create the user entry.")
        return None, ""

    # For each sheet, read the row once, check the new balance for each relevant unit
    for sheet_obj, unit_dict in sheet_costs.items():

        row_values = sheet_obj.row_values(row_index)

        # check for each unit
        for unit, qty in unit_dict.items():
            col_index = SHEET_COLUMNS.get(unit)
            if not col_index:
                await send(ctx, f"{unit} not recognized. Skipping it.")
            old_val_str = "0"
            if col_index <= len(row_values):
                old_val_str = row_values[col_index - 1]
            try:
                old_val_int = int(old_val_str)
            except ValueError:
                old_val_int = 0

            new_val = old_val_int - qty
            # build summary line
            cost_summary_lines.append(f"{unit}: {old_val_int} -> {new_val}")

            # check negative
            if 0 <= qty and new_val < 0:
                insufficient.append(unit)


    # Prepare an embed or a message summarizing cost
    cost_message = (
        f"{extra_message}"
        f"{title}" + "\n".join(cost_summary_lines)
    )

    # If we have any resources going negative, warn the user
    if insufficient:
        # Build a warning embed or text
        embed = discord.Embed(
            title = f"Insufficient Resources - {'Cancelling Action.'*int(auto_cancel)+'Proceed?'*int(not auto_cancel)}",
            description=(
                f"{member.display_name} does not have enough resources for this action without going into debt:\n"
                f"Would go negative in: {', '.join(insufficient)}\n\n"
                f"{"React with ✅ to confirm the action anyway, or ❌ to cancel."*int(not auto_cancel)+
                "Cancelling action."*int(auto_cancel)}"
            ),
            color=discord.Color.red()
        )
        embed.add_field(name="Cost Details", value=cost_message, inline=False)
        msg = await send(ctx, embed=embed)

        if auto_cancel: return None, ""

        # Add reaction emojis
        await msg.add_reaction("✅")
        await msg.add_reaction("❌")

        def check(reaction: Reaction, user: discord.Member):
            return (
                user == author(ctx)
                and str(reaction.emoji) in ["✅", "❌"]
                and reaction.message.id == msg.id
            )

        try:
            reaction, user = await bot.wait_for("reaction_add", timeout=60.0, check=check)
        except asyncio.TimeoutError:
            await send(ctx, "Action timed out. No changes were made.")
            return None, ""

        if str(reaction.emoji) == "❌":
            await send(ctx, "Action canceled.")
            return None, ""
        # If ✅, proceed with deduction
        else:
            return True, cost_message
    else:
        # No resources go negative, just show a confirmation
        await send(ctx, 
            embed=discord.Embed(
                title="Resources Sufficient",
                description=(
                    f"{member.display_name} has enough resources for this action:\n\n{cost_message}\n"
                    "Deducting resources now..."
                ),
                color=discord.Color.green()
            )
        )
        return True, cost_message

async def check_attachment(ctx: Context, channel_id: int, channel_name: str, msg_link: str) -> Optional[Tuple[Attachment, Union[GuildChannel, Thread, PrivateChannel]]]:
    # Ensure the channel is configured
    channel_id = config.get(f"{channel_name}_channel_id")
    if not channel_id:
        await send(ctx, f"The {channel_name} channel has not been configured. Use `!config {channel_name}_channel` to set it.")
        return None, None

    # Get the channel
    channel = bot.get_channel(channel_id)
    if not channel:
        await send(ctx, f"The configured {channel_name} channel is invalid or not accessible.")
        return None, None

    specified_msg = bool(msg_link)
    # Check if the message is a reply
    if not (ctx.message.reference or specified_msg):
        await send(ctx, "You need to reply to (or specifiy) a message with an attached image to use this command.")
        return None, channel

    # Get the referenced message
    if specified_msg:
        referenced_message  = await ctx.channel.fetch_message(int(msg_link.split('/')[6]))
    elif ctx.message.reference:
        referenced_message = await ctx.channel.fetch_message(ctx.message.reference.message_id)

    # Check if the referenced message has an attachment
    if not referenced_message.attachments:
        await send(ctx, "The replied message must contain an image attachment.")
        return None, channel

    # Get the first attachment (image)
    attachment = referenced_message.attachments[0]
    if not attachment.content_type.startswith("image"):
        await send(ctx, "The attachment must be an image.")
        return None, channel

    return attachment, channel, referenced_message

def get_member(member: discord.Member):

    sheet = main_sheet

    try:
        # Get all values in the column for user IDs
        user_ids = sheet.col_values(SHEET_COLUMNS[USER_ID])  
        user_row = user_ids.index(str(member.id)) + 1  # Convert index to 1-based row number
    except:
        return member

    record = sheet.cell(user_row, SHEET_COLUMNS[UNION_LEADER_ID]).value

    if record:
        user = bot.get_user(int(record))
        return user or member
    return member

def construct_unit(member: discord.Member, date: str, amount: int, type: unit_type, source: str):
    row = [
        str(member.id),      # UserID
        member.name,    # Discord name
        date,           # Date
        amount,         # Amount
        type,           # Type
        source,         # Source
    ]

    deployment_sheet.append_row(row)

def loan_unit(receiver: discord.Member, sender: discord.Member, date_of_arrival: str, date_of_return: str, amount: int, type: unit_type, source: str):
    row1 = [
        str(receiver.id), # UserID
        receiver.name,    # Discord name
        date_of_arrival,  # Date
        amount,           # Amount
        type,             # Type
        source,           # Source
        str(sender.id),   # UserID
        sender.name,      # Discord name
    ]

    row2 = [
        str(sender.id),   # UserID
        sender.name,      # Discord name
        date_of_return,   # Date
        amount,           # Amount
        type,             # Type
        source,           # Source
        str(receiver.id), # UserID
        receiver.name,    # Discord name
    ]

    mercenaries_sheet.append_rows([row1, row2])



async def add_log(ctx: Context, member: discord.Member, amount: str, unit: unit_type, log_msg: str):
    """Adds a unit and logs the transaction."""

    user_id = str(member.id)
    current_balance = add_user_balance(user_id , member.name, member.display_name, amount, unit)
    if current_balance is None:
        return await ctx.send(f"An error occurred when fetching the current {unit} balance of {member.display_name}.")

    log_transaction(
        user_id=user_id,
        user_name=member.name,
        change_amount=amount,
        unit=unit,
        source=log_msg,
        editor_id=ctx.author.id,
        editor_name=ctx.author.name,
        result=current_balance,
        message_link=ctx.message.jump_url
    )

    return current_balance



def batch_add_all(ctx: Context, member: discord.Member, resource_list: str, log_msg: str, mult: int=1):
    """
    Batch-adds multiple units to a single user's balance with minimal sheet writes,
    reading each relevant sheet only once and applying changes in one batch.
    Then logs each changed unit in a single batch.

    :param ctx: The command context (for logging, message URL).
    :param member: The Discord member whose balance is updated.
    :param resource_list: A list of (qty, unit) pairs, e.g. [(100, "Silver"), (2, "Crops")].
    :param log_msg: A source/reason message for logging.
    :param mult: A multiplier to apply to all qty in resource_list (defaults to 1).
    :return: A tuple (old_values, final_values)   # dictionaries of {unit: str_val}
    """

    # We'll accumulate total changes for each unit
    changes_map = {}  # unit -> total quantity
    for (qty, unit) in resource_list:
        changes_map[unit] = changes_map.get(unit, 0) + qty*mult

    # 2) Group changes by which sheet they belong to, so we can do minimal reads/writes
    #    e.g. { <sheet_obj>: { <unit>: qty, ... }, ... }
    sheet_changes: Dict[gspread.Worksheet, Dict[str, int]] = {}
    for unit, qty in changes_map.items():
        sheet_obj = get_sheet(unit)
        if not sheet_obj:
            continue
        if sheet_obj not in sheet_changes:
            sheet_changes[sheet_obj] = {}
        sheet_changes[sheet_obj][unit] = qty

    # 3) For each sheet, we read it once, find the row for user, 
    #    compute new values for each relevant unit, do one batch update
    # We'll also prepare log entries
    user_id_str = str(member.id)
    user_name = member.name
    display_name = member.display_name

    # We'll store for each changed resource, the final new value to log
    # e.g. final_values[unit] = new_balance
    final_values = {}
    old_values = {}

    # read all user IDs
    try:
        user_ids = next(iter(sheet_changes)).col_values(SHEET_COLUMNS[USER_ID])
        # find row
        row_index = user_ids.index(user_id_str) + 1
    except ValueError:
        return False

    for sheet_obj, unit_dict in sheet_changes.items():

        # We'll do partial batch update for each changed unit
        # but let's do it in a single request
        # We also need to read the old values to compute the final
        requests = []
        # We also want to store the final numeric value in final_values
        # We'll read the row in memory. Or do a single row_values?
        row_values = sheet_obj.row_values(row_index)  # entire row
        # We'll invert column->value from row_values
        # But let's do a simpler approach: for each changed unit, read the oldVal, parse int if possible
        # then newVal = oldVal + changes
        # add request in a single array
        has_name_updated = False

        for unit, delta in unit_dict.items():
            col_index = SHEET_COLUMNS.get(unit)
            if not col_index:
                continue
            old_val = "0"
            if col_index <= len(row_values):
                old_val = row_values[col_index - 1]  # col_index is 1-based, row_values is 0-based
            try:
                old_val_int = int(old_val)
            except ValueError:
                old_val_int = 0
            new_val_int = old_val_int + delta
            new_val_str = str(new_val_int)

            # queue up the batch update
            a1_notation = gspread.utils.rowcol_to_a1(row_index, col_index)
            requests.append({
                "range": a1_notation,
                "values": [[new_val_str]]
            })

            # if it's the silver_sheet, we also update user_name, display_name if needed
            # But you might do that in your existing set_user_balance logic. We'll do it here for batch:
            if not has_name_updated and sheet_obj == silver_sheet:
                has_name_updated = True
                # update discord name, display name
                dname_col = SHEET_COLUMNS.get(DISCORD_NAME)
                disp_col  = SHEET_COLUMNS.get(DISPLAY_NAME)
                if dname_col:
                    a1_dname = gspread.utils.rowcol_to_a1(row_index, dname_col)
                    requests.append({
                        "range": a1_dname,
                        "values": [[user_name]]
                    })
                if disp_col:
                    a1_disp = gspread.utils.rowcol_to_a1(row_index, disp_col)
                    requests.append({
                        "range": a1_disp,
                        "values": [[display_name]]
                    })

            # store final
            final_values[unit] = new_val_str
            old_values[unit] = old_val


        # do the batch update
        if requests:
            sheet_obj.batch_update(requests, value_input_option="USER_ENTERED")

    # 4) Once all sheets are updated in one pass each, we log each resource separately in one batch
    # build the logs
    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    editor_id = str(ctx.author.id)
    editor_name = ctx.author.name

    log_rows = []
    for (qty, unit) in resource_list:
        # final new val is final_values.get(unit)
        final_val_str = final_values.get(unit, "N/A")

        row = [
            user_id_str,          # user ID
            user_name,            # Discord name
            timestamp_str,        # Date
            str(qty),             # Amount changed
            unit,                 # Type
            log_msg,              # Source
            editor_id,            # EditorID
            editor_name,          # Editor Discord name
            final_val_str,        # Result
            ctx.message.jump_url  # Message Link
        ]
        log_rows.append(row)

    if log_rows:
        log_sheet.append_rows(log_rows, value_input_option="USER_ENTERED")

    return (old_values, final_values)
    # done, send confirmation
    #await ctx.send(f"Batch add for {member.display_name} completed. Updated {len(resource_list)} resources in minimal sheet writes!")

@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f'Bot is ready. Logged in as {bot.user}.')
@bot.event
async def on_command_error(ctx: Context, error):
    if isinstance(error, commands.CommandNotFound):
        # Command doesn't exist
        # Show a generic help message with the list of commands
        await send(ctx, "Command not found. Use `!help` to see the list of available commands.", ephemeral=True)
    elif isinstance(error, commands.MissingRequiredArgument):
        # Missing arguments for command
        # Show command usage from command.help
        cmd = ctx.command
        if cmd and cmd.help:
            await send(ctx, f"Missing arguments.\n**Usage:**\n```{ctx.prefix}{cmd.name} {cmd.help}```", ephemeral=True)
        else:
            await send(ctx, "You are missing required arguments for this command.")
    elif isinstance(error, commands.BadArgument):
        # Bad argument type
        cmd = ctx.command
        if cmd and cmd.help:
            await send(ctx, f"Invalid argument.\n**Usage:**\n```{ctx.prefix}{cmd.name} {cmd.help}```", ephemeral=True)
        else:
            await send(ctx, "Invalid argument provided.", ephemeral=True)
    elif isinstance(error, commands.CheckFailure):
        # User is not authorized or does not meet a check
        await send(ctx, "You do not have permission to use this command.", ephemeral=True)
    else:
        # Other errors, just print to console and notify user
        print(error)
        await send(ctx, "An error occurred. Check the console for more details.", ephemeral=True)

# -------------- COMMANDS --------------

# Basic functions


@bot.command(name="sync", brief="Syncs the command tree.")
async def sync(ctx: commands.Context):
    """
    Syncs the app command tree. Only used by admins.
    """

    # Check if the command issuer is authorized
    if not ctx.author.guild_permissions.administrator:
        return await ctx.send("You do not have permission to configure the bot. Only Admins are allowed.")

    await bot.tree.sync()
    return await ctx.send("Command Tree synced.")


@bot.hybrid_command(name="create", brief="Create a player entry.")
@app_commands.describe(
    member="The Discord member to create the entry for.",
    nation_name="The name of the player's nation.",
    capital="The capital tile of the player's nation.",
    religion="The religion of the player's nation (optional).",
    culture="The culture of the player's nation (optional).",
    subculture="The subculture of the player's nation (optional).",
    xp="The initial XP for the player (default is 0).",
    silver="The initial silver balance for the player (default is 0).",
    tiles="The initial number of tiles for the player (default is 1).",
    status="The player's initial status (default is 'Independent')."
)
async def create(ctx: commands.Context,
                 member: discord.Member,

                 nation_name: str, 
                 capital: str, 

                 religion: str="", 
                 culture:str="",

                 silver: int=0, 
                 tiles: int=12):
    """
    Create a player entry in the game sheet, initializing their stats across all sheets.\n
    ‎\n
    Usage\n
    -----------\n
    !create @User "Nation Name" "Capital" [Religion] [Culture] [Subculture] [XP] [Silver] [Tiles] [Status]\n
    ‎\n
    Example\n
    -----------\n
    !create @User "Empire of France" "Paris"\n

    Parameters
    -----------
    member: discord.Member
        The Discord member to create the entry for.
    nation_name: str
        The name of the player's nation.
    capital: str
        The capital of the player's nation.
    religion: str, optional
        The religion of the player's nation (default is "").
    culture: str, optional
        The culture of the player's nation (default is "").
    silver: int, optional
        The initial silver balance for the player (default is 0).
    tiles: int, optional
        The initial number of tiles for the player (default is 12).
    """
    if ctx.interaction and not ctx.interaction.response.is_done():
        await ctx.interaction.response.defer()  # Acknowledge the interaction immediately

    # Check if the command issuer is authorized
    if not is_authorized(ctx):
        return await ctx.send("You do not have permission to create player entries.")

    # Check if user exists
    user_name = str(member.name)

    user_names = main_sheet.col_values(SHEET_COLUMNS[DISCORD_NAME])  # Fetch all user IDs from the sheet
    if str(user_name) in user_names:
        return await ctx.send(f"User {member.display_name} ({user_name}) already created.")

    # Column order:
    main_row = [
        user_name,          # Discord Name 
        nation_name,        # Nation name
        "",                 # Region
        religion,           # Religion
        culture,            # Culture
        "",                 # Title
        capital,            # Capital
        1,                  # Cities
        int(tiles),         # Tiles
        int(silver),        # Silver
        0,                  # Food
        0,                  # Fuel
        0,                  # Gems
        0,                  # Metal
        0,                  # Stone
        0,                  # Timber
        0                   # Spawns used
    ]

    buildings_row = [
        user_name,
        nation_name,
        0,  # Tier 1 City Count
        1,  # Tier 2 City Count
        0,  # Tier 3 City Count
        3,  # Brigade Cap
        1,  # General Cap
        0,  # Ship Cap, default 0, 1 if on coast


    ]

    main_sheet.append_row(main_row)
    buildings_sheet.append_row(buildings_row)


    await ctx.send(f"Created initial player entry for {member.display_name} ({user_name}).")


@bot.hybrid_command(name="set", brief="Set a user's property.")
@app_commands.describe(
    member="The Discord member whose property to set.",
    value="The value to set for the unit.",
    unit_str="The unit type (The column name in the game sheet e.g., Silver, Food, Tiles, Brigade Cap)."
)
async def set(ctx: commands.Context, member: discord.Member, value: str, unit_str: str):
    """
    Set a user's property in the game sheet.\n
    ‎\n
    Usage\n
    -----------\n
    !set @User value unit [source]\n
    ‎\n
    Example\n
    -----------\n
    !set @User 100 Silver "Starting Silver"\n

    Parameters
    -----------
    member: discord.Member
        The Discord member whose property you want to set.
    value: str
        The value to set for the unit (e.g., 100).
    unit_str: str
        The unit type to set (e.g., "Silver", "Tiles", "Food").
    """

    if ctx.interaction and not ctx.interaction.response.is_done():
        await ctx.interaction.response.defer()  # Acknowledge the interaction immediately

    if not is_authorized(ctx):
        return await ctx.send("You do not have permission to modify the sheet.")

    unit = get_unit(unit_str) 
    if not unit:
        return await ctx.send(f"{unit_str} is not a valid unit. Please input valid unit.")
    member = get_member(member)
    if member == None:
        return await ctx.send(f"Member fetch error.")
    user_name = str(member.name)
    new_value = set_user_balance(user_name, value, unit)
    if new_value == None:
        return await ctx.send(f"User {member.display_name} ({user_name}) does not exist. Use !create to create the user entry.")

    if value.isnumeric():
        value = int(value)
    await ctx.send(f"Set {member.display_name}'s {unit} to {value} {unit}.")


@bot.hybrid_command(name="add", brief="Add an amount to a user's chosen unit.")
@app_commands.describe(
    member="The Discord member to whom the amount will be added.",
    amount="The amount to add to the unit.",
    unit_str="The unit type (The column name in the game sheet e.g., Silver, XP, Crops, Status, Army)."
)
async def add(ctx: Context, member: discord.Member, amount: int, unit_str: str):
    """
    Add an amount of a chosen unit to a user's balance.\n
    ‎\n
    Usage\n
    -----------\n
    !add @User amount unit [source] [add_xp]\n
    ‎\n 
    Examples\n
    -----------\n
    !add @User 100 Silver "For brilliant essay!"\n
    !add @User 10 Silver "Xp-less gift" False\n

    Parameters
    -----------
    member: discord.Member
        The member to whom the amount will be added.
    amount: int
        The amount to add to the unit.
    unit_str: str
        The unit type (e.g., "Silver", "XP", "Crops").
    source: str, optional
        The reason for the addition (default is "No source provided").
    add_xp: bool, optional
        Whether to auto-add the same amount of XP if the unit is Silver (default is True).
    """
    if ctx.interaction and not ctx.interaction.response.is_done():
        await ctx.interaction.response.defer()  # Acknowledge the interaction immediately

    if not is_authorized(ctx):
        return await send(ctx, "You do not have permission to modify the sheet.")

    unit = get_unit(unit_str) 
    if not unit:
        return await send(ctx, f"{unit_str} is not a valid unit. Please input valid unit.")

    member = get_member(member)
    if member == None:
        return await ctx.send(f"Member fetch error.")
    user_name = str(member.name)


    result = add_user_balance(user_name, amount, unit)
    if result == None:
        return await ctx.send(f"User {member.display_name} ({user_name}) does not exist. Use !create to create the user entry.")

    return await send(ctx, f"Added {amount} {unit} to {member.display_name}.")


@bot.hybrid_command(name="add_all", brief="Add multiple resources to a player's balance in one batch.")
@app_commands.describe(
    member="The user to whom these resources will be added.",
    resources='A comma-separated list of resources (e.g., "100 Silver, -2 Crops").',
)
async def add_all(
    ctx: commands.Context, 
    member: discord.Member, 
    resources: str, 
    source: str = "No source provided"
):
    """
    Add multiple resources to a single player's balance in one minimal sheet update.

    Usage
    ------
      !add_all @User "100 Silver, -2 Crops"

    Example
    --------
      !add_all @User "10 Silver, 5 XP" "Quest Reward"

    Parameters
    -----------
    member: discord.Member
        The user who will receive these resources.
    resources: str
        A comma-separated list of resource additions (e.g., "100 Silver, -2 Crops").
    source: str, optional
        A short message or reason for the transaction, default is "No source provided".
    """
    if ctx.interaction and not ctx.interaction.response.is_done():
        await ctx.interaction.response.defer()  # Acknowledge the interaction immediately

    if not is_authorized(ctx):
        return await send(ctx, "You do not have permission to modify the sheet.")

    member = get_member(member)
    if member == None:
        return await ctx.send(f"Member fetch error.")

    resource_list, invalid = parse_resource_list(resources)

    if invalid:
        return await ctx.send(f"Invalid resources: {', '.join(invalid)}. Aborting Addition.")

    if not resource_list:
        return await ctx.send("No valid resources to add.")
        
    resource_list = group_costs(resource_list)


    confirm = batch_add_all(ctx, member, resource_list, source)
    if not confirm:
        return await ctx.send("User not found.")
    old, final = confirm
    
    cost_summary_lines = []
    
    for unit in final.keys():
        cost_summary_lines.append(
            f"{unit}: {old[unit]} -> {final[unit]}"
        )

    # Prepare an embed or a message summarizing cost
    cost_message = (
        "**Change Summary**\n" + "\n".join(cost_summary_lines)
    )

    return await send(ctx, 
        embed=discord.Embed(
            description=(
                f"Added resources to {member.display_name}. New Amounts:\n\n{cost_message}\n"
            ),
            color=discord.Color.blue()
        )
    )



@bot.hybrid_command(name="delete", brief="Delete a user's records from all sheets except the log.")
@app_commands.describe(
    member="The Discord member whose records will be deleted from all sheets."
)
async def delete_command(ctx: commands.Context, member: discord.Member):
    """
    Deletes all records of the specified user from the main sheets (View, Resource, Production, Buildings, etc.), 
    leaving no empty rows behind. Does not remove entries from the log sheet.\n 
    ‎\n 
    Example\n 
    -----------\n 
    !delete @User

    Parameters
    -----------
    member: discord.Member
        The Discord member whose records will be deleted.
    """
    if ctx.interaction and not ctx.interaction.response.is_done():
        await ctx.interaction.response.defer()  # Acknowledge the interaction immediately

    if not is_authorized(ctx):
        return await ctx.send("You do not have permission to delete player entries.")

    sheets = {
        f"{MAIN_SHEET} Sheet": main_sheet,
        f"{BUILDINGS_SHEET} Sheet": buildings_sheet
    }

    user_name = member.name  # Convert user ID to a string once
    deleted_sheets = []

    for sheet_name, sheet_obj in sheets.items():
        try:
            user_names = sheet_obj.col_values(SHEET_COLUMNS[DISCORD_NAME])  # Fetch all UserID values
            if user_name in user_names:
                row_index = user_names.index(user_name) + 1  # Convert to 1-based index
                sheet_obj.delete_rows(row_index)  # Delete the row
                deleted_sheets.append(sheet_name)
        except Exception as e:
            # Handle unexpected errors and continue
            print(f"Error processing {sheet_name}: {e}")

    if not deleted_sheets:
        await ctx.send(f"No records found for {member.display_name} in any of the main sheets.")
    else:
        await ctx.send(
            f"Deleted {member.display_name}'s entry from the following sheets: {', '.join(deleted_sheets)}."
        )

# Logging and showing

#@bot.command(name="mystatus", brief="Show your current info or a specific unit's value.")
async def mystatus(ctx: commands.Context, unit_str: str | None = None):
    """
    [unit]
    Show the user's current information.
    - If a unit is specified (e.g. !mystatus silver), shows that unit's value.
    - If no unit is specified (just !mystatus), show all user data.
    """
    if isinstance(ctx.interaction, Interaction):
        if not ctx.interaction.response.is_done():
            await ctx.interaction.response.defer()  # Acknowledge the interaction immediately

    user_name = ctx.author.name

    if unit_str:
        # If a unit is specified, show just that unit
        unit = get_unit(unit_str)
        if not unit:
            return await ctx.send(f"{unit_str} is not a valid unit. Please input a valid unit.")

        balance = get_user_balance(user_name, unit)
        if balance is None:
            return await ctx.send("You have no entry yet or the specified unit is unavailable.")

        embed = discord.Embed(title=f"Your {unit} Status", color=discord.Color.gold())
        embed.add_field(name="Balance", value=str(balance), inline=False)
        embed.set_thumbnail(url=ctx.author.avatar.url if ctx.author.avatar else "")
        embed.set_footer(text=f"{unit} status")
        return await ctx.send(embed=embed)
    else:
        # No unit specified, show all user info
        try:
            # Find user rows in each sheet
            user_row_main = main_sheet.find(user_name).row
        except:
            return await ctx.send("You don't have any entry in the database yet. Please ask a mod to create your entry.")

        # Retrieve data from silver_sheet (View sheet)
        # Using SHEET_COLUMNS keys that map to that sheet
        def val(sheet: Worksheet, key):
            # Safely retrieve value by key if exists
            col = SHEET_COLUMNS.get(key)
            return sheet.cell(user_row_main, col).value if col else None

        discord_name = val(main_sheet, "Discord Name")
        nation_name = val(main_sheet, "Nation Name")
        region = val(main_sheet, "Region")
        capital = val(main_sheet, "Capital")
        silver_val = val(main_sheet, "Silver")
        tiles = val(main_sheet, "Tile Count")
        culture = val(main_sheet, "Culture")
        religion = val(main_sheet, "Religion")
        title = val(main_sheet, "Title")
        cities = val(main_sheet, "Cities")
        food = val(main_sheet, "Food")
        fuel = val(main_sheet, "Fuel")
        gems = val(main_sheet, "Gems")
        metal = val(main_sheet, "Metal")
        stone = val(main_sheet, "Stone")
        timber = val(main_sheet, "Timber")


        # Try fetching Resource and Production data
        # We assume user_id also exists there; if not, we handle gracefully
        try:
            user_row_buildings = buildings_sheet.find(user_name).row
        except:
            user_row_buildings = None

        def val_building(key):
            if not user_row_buildings: return "N/A"
            col = SHEET_COLUMNS.get(key)
            return buildings_sheet.cell(user_row_buildings, col).value if col else "N/A"

        # Buildings

        t1_cities = val(buildings_sheet, "T1 City Count")
        t2_cities = val(buildings_sheet, "T2 City Count")
        t3_cities = val(buildings_sheet, "T3 City Count")
        brigade_cap = val(buildings_sheet, "Brigade Count")
        ship_cap = val(buildings_sheet, "Ship Count")
        general_cap = val(buildings_sheet, "General Count")

        embed = discord.Embed(title=f"{ctx.author.display_name}'s Status", color=discord.Color.blue())
        embed.set_thumbnail(url=ctx.author.avatar.url if ctx.author.avatar else "")

        # Basic info field
        embed.add_field(name="Basic Info", value=(
            f"**Discord Name:** {discord_name}\n"
            f"**Nation Name:** {nation_name}\n"
            f"**Region:** {region}\n"
            f"**Religion:** {religion}\n"
            f"**Culture:** {culture}\n"
            f"**Title:** {title}\n"
            f"**Capital:** {capital}\n"
            f"**Cities:** {cities}\n"
            f"---Tier 1:-- {t1_cities}\n"
            f"---Tier 2:-- {t2_cities}\n"
            f"---Tier 3:-- {t3_cities}\n"
            f"**Tiles:** {tiles}\n"
            f"**Silver:** {silver_val}\n"
            f"**Military:**\n"
            f"---Brigade Cap:-- {brigade_cap}\n"
            f"---Ship Cap:-- {ship_cap}\n"
            f"---General Cap:-- {general_cap}\n"
        ), inline=False)
        embed.set_footer(text="Overall Status")
        await ctx.send(embed=embed)


VALID_CATEGORIES = ["buildings","main"]

async def category_autocomplete(
    ctx: Context,
    current: str,
) -> List[app_commands.Choice[str]]:
    return [
        app_commands.Choice(name=cat.capitalize(), value=cat)
        for cat in VALID_CATEGORIES if current.lower() in cat.lower()
    ]

@bot.hybrid_command(name="status", brief="Show a player's info of a specific category or unit.")
@app_commands.describe(
    category="The category or unit to display (e.g., resources, buildings, production, war, view; or Silver, Crops, XP, Status).",
    member="The member whose status is being queried. Defaults to the command issuer if not specified."
)
@app_commands.autocomplete(category=category_autocomplete)
async def status(ctx: commands.Context, category: str | None = None, member: discord.Member = None):
    """
    Show a player's current information or a specific category.\n 
    - If a category is specified (e.g., !status resources @User), shows that category.\n 
    - If no category is specified, the entire status is shown.\n 
    - If no user is specified, shows the author's status.\n 
    ‎\n 
    Usage\n 
    -----------\n 
    !status [category] [@member]\n 

    Parameters
    -----------
    category: str, optional
        The category to display. It can be a sheet (e.g., resources, buildings, production, war, view) or a specific unit (e.g. e.g., Silver, XP, Crops, Status, Army). If omitted, displays all info.
    member: discord.Member, optional
        The member whose status is being queried. Defaults to the command issuer if not specified.
    """
    if ctx.interaction and not ctx.interaction.response.is_done():
        await ctx.interaction.response.defer()   # Acknowledge the interaction immediately


    # Default to the command author if no member is provided
    if not member:
        member = ctx.author
    elif ctx.author != member and not is_authorized(ctx):
        return await ctx.send("You do not have permission to check another player's status.")

    member = get_member(member)
    user_id = member.id

    try:
        # Get all values in the column for user IDs
        user_ids = silver_sheet.col_values(SHEET_COLUMNS[USER_ID])  
        user_row = user_ids.index(str(user_id)) + 1  # Convert index to 1-based row number
    except:
        return await ctx.send(f"User {member.display_name} does not exist. Use !create to create the user entry.")

    # Check if the category or unit is specified
    if category:
        category = category.lower()
        if category not in VALID_CATEGORIES:
            unit = get_unit(category)
            if not unit:
                return await ctx.send(
                    f"Invalid category '{category}'. Valid categories: {', '.join(VALID_CATEGORIES)}."
                )
            else:
                sheet = get_sheet(unit)
                balance = sheet.cell(user_row, SHEET_COLUMNS[unit]).value

                embed = discord.Embed(title=f"{member.display_name}'s {unit} Status", color=discord.Color.gold())
                embed.add_field(name="Balance", value=str(balance), inline=False)
                embed.set_thumbnail(url=member.avatar.url if member.avatar else "")
                embed.set_footer(text=f"{unit} status")
                return await ctx.send(embed=embed)

        elif category == "resources":

            row = resource_sheet.row_values(user_row)
            def val_resource(key):
                col = SHEET_COLUMNS.get(key)-1
                if col >= len(row):
                    return "N/A"
                return row[col]

            resource_stats = {
                "Crops": val_resource(CROPS),
                "Fuel": val_resource(FUEL),
                "Stone": val_resource(STONE),
                "Timber": val_resource(TIMBER),
                "Livestock": val_resource(LIVESTOCK),
                "Mounts": val_resource(MOUNTS),
                "Metal": val_resource(METAL),
                "Fiber": val_resource(FIBER),
                "Industry": val_resource(INDUSTRY),
                "Energy": val_resource(ENERGY),
                "Tools": val_resource(TOOLS),
                "Cement": val_resource(CEMENT),
                "Supplies": val_resource(SUPPLIES),
            }

            resource_str = "\n".join([f"**{r}:** {val}" for r, val in resource_stats.items()])
            embed = discord.Embed(
                title=f"{member.display_name}'s Resources",
                description=resource_str,
                color=discord.Color.blue()
            )
            embed.set_thumbnail(url=member.avatar.url if member.avatar else "")
            return await ctx.send(embed=embed)

        elif category == "buildings":

            row = buildings_sheet.row_values(user_row)
            def val_building(key):
                col = SHEET_COLUMNS.get(key)-1
                if col >= len(row):
                    return "N/A"
                return row[col]

            building_stats = {
                "T1City": val_building(T1_CITY),
                "T2City": val_building(T2_CITY),
                "T3City": val_building(T3_CITY),
                "T1Industry": val_building(T1_INDUSTRY),
                "T2Industry": val_building(T2_INDUSTRY),
                "T3Industry": val_building(T3_INDUSTRY),
                "T1Fort": val_building(T1_FORT),
                "T2Fort": val_building(T2_FORT),
                "T3Fort": val_building(T3_FORT),
                "Monument": val_building(MONUMENT),
            }

            buildings_str = "\n".join([f"**{b}:** {val}" for b, val in building_stats.items()])
            embed = discord.Embed(
                title=f"{member.display_name}'s Buildings",
                description=buildings_str,
                color=discord.Color.blue()
            )
            embed.set_thumbnail(url=member.avatar.url if member.avatar else "")
            return await ctx.send(embed=embed)

        elif category == "production":

            row = production_sheet.row_values(user_row)
            def val_production(key):
                col = SHEET_COLUMNS.get(key)-1
                if col >= len(row):
                    return "N/A"
                return row[col]

            production_stats = {
                "CropsTile": val_production(CROPS_TILE),
                "FuelTile": val_production(FUEL_TILE),
                "StoneTile": val_production(STONE_TILE),
                "TimberTile": val_production(TIMBER_TILE),
                "LivestockTile": val_production(LIVESTOCK_TILE),
                "MountsTile": val_production(MOUNTS_TILE),
                "MetalTile": val_production(METAL_TILE),
                "FiberTile": val_production(FIBER_TILE),
            }

            production_str = "\n".join([f"**{p}:** {val}" for p, val in production_stats.items()])
            embed = discord.Embed(
                title=f"{member.display_name}'s Production Tiles",
                description=production_str,
                color=discord.Color.blue()
            )
            embed.set_thumbnail(url=member.avatar.url if member.avatar else "")
            return await ctx.send(embed=embed)

        elif category == "view":

            row = silver_sheet.row_values(user_row)
            def val_silver(key):
                col = SHEET_COLUMNS.get(key)-1
                if col >= len(row):
                    return "N/A"
                return row[col]


            discord_name = val_silver(DISCORD_NAME)
            nation_name = val_silver(NATION_NAME)
            capital = val_silver(CAPITAL)

            xp = val_silver(XP)
            silver_val = val_silver(SILVER)
            tiles = val_silver(TILES)

            culture = val_silver(CULTURE)
            subculture = val_silver(SUBCULTURE)
            religion = val_silver(RELIGION)


            embed = discord.Embed(
                title=f"{member.display_name}'s Basic Info",
                color=discord.Color.blue()
            )
            embed.set_thumbnail(url=member.avatar.url if member.avatar else "")
            embed.add_field(name="Basic Info", value=(
                f"**Discord Name:** {discord_name}\n"
                f"**Nation Name:** {nation_name}\n"
                f"**Capital:** {capital}\n"

                f"**XP:** {xp}\n"
                f"**Silver:** {silver_val}\n"
                f"**Tiles:** {tiles}\n"

                f"**Culture:** {culture}\n"
                f"**Subculture:** {subculture}\n"
                f"**Religion:** {religion}\n"
            ), inline=False)
            return await ctx.send(embed=embed)   

        elif category == "war":

            row = war_sheet.row_values(user_row)
            def val_war(key):
                col = SHEET_COLUMNS.get(key)-1
                if col >= len(row):
                    return "N/A"
                return row[col]


            army = val_war(ARMY)
            army_cap = val_war(ARMY_CAP)
            navy = val_war(NAVY)
            navy_cap = val_war(NAVY_CAP)

            army_doc = val_war(ARMY_DOCTRINE)
            navy_doc = val_war(NAVY_DOCTRINE)

            temp_army = val_war(TEMP_ARMY)
            temp_navy = val_war(TEMP_NAVY)

            capital_navy = val_war(CAPITAL_NAVY)
            general = val_war(GENERAL)

            embed = discord.Embed(
                title=f"{member.display_name}'s Basic Info",
                color=discord.Color.blue()
            )
            embed.set_thumbnail(url=member.avatar.url if member.avatar else "")
            embed.add_field(name="Basic Info", value=(
                f"**Army:** {army}+{temp_army}/{army_cap}\n"
                f"**Navy:** {navy}+{temp_navy}/{navy_cap}\n"
                f"**Army Doctrine:** {army_doc}\n"
                f"**Navy Doctrine:** {navy_doc}\n"
                f"{f"**Capital Navy Bonus:** {capital_navy}\n" if capital_navy else ""}"
                f"{f"**General Bonus:** {general}\n" if capital_navy else ""}"
                
            ), inline=False)
            return await ctx.send(embed=embed)         
        else:
            return await ctx.send(f"Invalid category '{category}'. Valid categories: resources, buildings, production, war, view.")

    
    resource_row = resource_sheet.row_values(user_row)
    def val_resource(key):
        col = SHEET_COLUMNS.get(key)-1
        if col >= len(resource_row):
            return "N/A"
        return resource_row[col]

    buildings_row = buildings_sheet.row_values(user_row)
    def val_building(key):
        col = SHEET_COLUMNS.get(key)-1
        if col >= len(buildings_row):
            return "N/A"
        return buildings_row[col]

    production_row = production_sheet.row_values(user_row)
    def val_production(key):
        col = SHEET_COLUMNS.get(key)-1
        if col >= len(production_row):
            return "N/A"
        return production_row[col]

    silver_row = silver_sheet.row_values(user_row)
    def val_silver(key):
        col = SHEET_COLUMNS.get(key)-1
        if col >= len(silver_row):
            return "N/A"
        return silver_row[col]

    war_row = war_sheet.row_values(user_row)
    def val_war(key):
        col = SHEET_COLUMNS.get(key)-1
        if col >= len(war_row):
            return "N/A"
        return war_row[col]


    discord_name = val_silver(DISCORD_NAME)
    nation_name = val_silver(NATION_NAME)
    capital = val_silver(CAPITAL)

    xp = val_silver(XP)
    silver_val = val_silver(SILVER)
    tiles = val_silver(TILES)

    culture = val_silver(CULTURE)
    subculture = val_silver(SUBCULTURE)
    religion = val_silver(RELIGION)

    status = val_silver(STATUS)
    union_leader = val_silver(UNION_LEADER)
    title = val_silver(TITLE)
    other_bonuses = val_silver(OTHER_BONUSES)


    # Gather some key stats from resources
    resource_stats = {
        "Crops": val_resource(CROPS),
        "Fuel": val_resource(FUEL),
        "Stone": val_resource(STONE),
        "Timber": val_resource(TIMBER),
        "Livestock": val_resource(LIVESTOCK),
        "Mounts": val_resource(MOUNTS),
        "Metal": val_resource(METAL),
        "Fiber": val_resource(FIBER),
        "Industry": val_resource(INDUSTRY),
        "Energy": val_resource(ENERGY),
        "Tools": val_resource(TOOLS),
        "Cement": val_resource(CEMENT),
        "Supplies": val_resource(SUPPLIES),
    }

    # Production tiles (CropsTile, FuelTile, etc.) - these are on production_sheet
    production_stats = {
        "CropsTile": val_production(CROPS_TILE),
        "FuelTile": val_production(FUEL_TILE),
        "StoneTile": val_production(STONE_TILE),
        "TimberTile": val_production(TIMBER_TILE),
        "LivestockTile": val_production(LIVESTOCK_TILE),
        "MountsTile": val_production(MOUNTS_TILE),
        "MetalTile": val_production(METAL_TILE),
        "FiberTile": val_production(FIBER_TILE),
    }

    # Buildings

    building_stats = {
        "T1City": val_building(T1_CITY),
        "T2City": val_building(T2_CITY),
        "T3City": val_building(T3_CITY),
        "T1Industry": val_building(T1_INDUSTRY),
        "T2Industry": val_building(T2_INDUSTRY),
        "T3Industry": val_building(T3_INDUSTRY),
        "T1Fort": val_building(T1_FORT),
        "T2Fort": val_building(T2_FORT),
        "T3Fort": val_building(T3_FORT),
        "Monument": val_building(MONUMENT),
    }

    # War
    war_stats = {
        "Army": f"{val_war(ARMY)}+{val_war(TEMP_ARMY)}/{val_war(ARMY_CAP)}",
        "Navy": f"{val_war(NAVY)}+{val_war(TEMP_NAVY)}/{val_war(NAVY_CAP)}",
        "Army Doctrine": val_war(ARMY_DOCTRINE),
        "Navy Doctrine": val_war(NAVY_DOCTRINE),
        "Capital Navy Bonus": val_war(CAPITAL_NAVY),
        "General Bonus": val_war(GENERAL)
    }

    embed = discord.Embed(title=f"{member.display_name}'s Status", color=discord.Color.blue())
    embed.set_thumbnail(url=member.avatar.url if member.avatar else "")

    # Basic info field
    embed.add_field(name="Basic Info", value=(
        f"**Discord Name:** {discord_name}\n"
        f"**Nation Name:** {nation_name}\n"
        f"**Capital:** {capital}\n"
        f"**XP:** {xp}\n"
        f"**Silver:** {silver_val}\n"
        f"**Tiles:** {tiles}\n"
        f"**Culture:** {culture}\n"
        f"**Subculture:** {subculture}\n"
        f"**Religion:** {religion}\n"
        f"**Status:** {status} {f'(Union Leader: {union_leader})' if union_leader else ''}\n"
        f"**Title:** {title}\n"
        f"**Other Bonuses:** {other_bonuses}\n"
    ), inline=False)

    # Resources Field
    resource_str = "\n".join([f"**{r}:** {val}" for r, val in resource_stats.items()])
    embed.add_field(name="Resources", value=resource_str, inline=False)

    # Production Field
    production_str = "\n".join([f"**{p}:** {val}" for p, val in production_stats.items()])
    embed.add_field(name="Production Tiles", value=production_str, inline=False)

    # Buildings Field
    buildings_str = "\n".join([f"**{b}:** {val}" for b, val in building_stats.items()])
    embed.add_field(name="Buildings", value=buildings_str, inline=False)

    # War Field
    war_str = "\n".join([f"**{b}:** {val}" for b, val in war_stats.items() if not (b=="Capital Navy Bonus" or b=="General Bonus") or str(val)]) # only include those 2 bonuses if they exist
    embed.add_field(name="War", value=war_str, inline=False)


    embed.set_footer(text="Overall Status")
    return await ctx.send(embed=embed)

"""
VALID_CATEGORIES = ["resources", "buildings", "production", "war", "view"]

async def category_autocomplete(
    ctx: Context,
    current: str,
) -> List[app_commands.Choice[str]]:
    return [
        app_commands.Choice(name=cat.capitalize(), value=cat)
        for cat in VALID_CATEGORIES if current.lower() in cat.lower()
    ]

@bot.hybrid_command(name="status", brief="Show a player's info of a specific category or unit.")
@app_commands.describe(
    category="The category or unit to display (e.g., resources, buildings, production, war, view; or Silver, Crops, XP, Status).",
    member="The member whose status is being queried. Defaults to the command issuer if not specified."
)
@app_commands.autocomplete(category=category_autocomplete)
async def status(ctx: commands.Context, category: str | None = None, member: discord.Member = None):
    ""
    Show a player's current information or a specific category.\n 
    - If a category is specified (e.g., !status resources @User), shows that category.\n 
    - If no category is specified, the entire status is shown.\n 
    - If no user is specified, shows the author's status.\n 
    ‎\n 
    Usage\n 
    -----------\n 
    !status [category] [@member]\n 

    Parameters
    -----------
    category: str, optional
        The category to display. It can be a sheet (e.g., resources, buildings, production, war, view) or a specific unit (e.g. e.g., Silver, XP, Crops, Status, Army). If omitted, displays all info.
    member: discord.Member, optional
        The member whose status is being queried. Defaults to the command issuer if not specified.
    ""
    if ctx.interaction and not ctx.interaction.response.is_done():
        await ctx.interaction.response.defer()

    # Default to the command author if no member is provided
    if not member:
        member = ctx.author
    elif ctx.author != member and not is_authorized(ctx):
        return await ctx.send("You do not have permission to check another player's status.")

    user_id = str(member.id)
    category = category.lower() if category else None

    # Define a helper function for retrieving cell values
    def get_value(sheet, row, key):
        col = SHEET_COLUMNS.get(key)
        return sheet.cell(row, col).value if col else "N/A"

    # Map categories to their corresponding sheets and data processors
    category_map = {
        "resources": (resource_sheet, lambda row: {
            key: get_value(resource_sheet, row, key)
            for key in SHEET_COLUMNS.keys() if key in resources
        }),
        "buildings": (buildings_sheet, lambda row: {
            key: get_value(buildings_sheet, row, key)
            for key in SHEET_COLUMNS.keys() if key in buildings
        }),
        "production": (production_sheet, lambda row: {
            key: get_value(production_sheet, row, key)
            for key in SHEET_COLUMNS.keys() if key in production_tiles
        }),
        "war": (war_sheet, lambda row: {
            "Army": f"{get_value(war_sheet, row, ARMY)}+{get_value(war_sheet, row, TEMP_ARMY)}/{get_value(war_sheet, row, ARMY_CAP)}",
            "Navy": f"{get_value(war_sheet, row, NAVY)}+{get_value(war_sheet, row, TEMP_NAVY)}/{get_value(war_sheet, row, NAVY_CAP)}",
            "Army Doctrine": get_value(war_sheet, row, ARMY_DOCTRINE),
            "Navy Doctrine": get_value(war_sheet, row, NAVY_DOCTRINE)
        }), 
        "view": (silver_sheet, lambda row: {
            key: get_value(silver_sheet, row, key)
            for key in SHEET_COLUMNS.keys() if key in mix
        }),
    }

    # If a category is specified, handle it
    if category:
        if category not in category_map:
            unit = get_unit(category)
            if not unit:
                return await ctx.send(
                    f"Invalid category '{category}'. Valid categories: {', '.join(category_map.keys())}."
                )
            else:
                if unit:
                    # If a unit is specified, show just that unit
                    if not unit:
                        return await ctx.send(f"{category} is not a valid unit. Please input a valid unit.")

                    balance = get_user_balance(user_id, unit)
                    if balance is None:
                        return await ctx.send(f"{member.display_name} does not have an entry yet or the specified unit is unavailable.")

                    embed = discord.Embed(title=f"{member.display_name}'s {unit} Status", color=discord.Color.gold())
                    embed.add_field(name="Balance", value=str(balance), inline=False)
                    embed.set_thumbnail(url=member.avatar.url if member.avatar else "")
                    embed.set_footer(text=f"{unit} status")
                    return await ctx.send(embed=embed)

        sheet, processor = category_map[category]
        try:
            user_row = sheet.find(user_id).row
        except gspread.exceptions.CellNotFound:
            return await ctx.send(f"{member.display_name} does not have data in the {category} category.")

        data = processor(user_row)
        formatted_data = "\n".join([f"**{key}:** {value}" for key, value in data.items()])
        embed = discord.Embed(
            title=f"{member.display_name}'s {category.capitalize()}",
            description=formatted_data,
            color=discord.Color.blue()
        )
        embed.set_thumbnail(url=member.avatar.url if member.avatar else "")
        return await ctx.send(embed=embed)

    # No category specified, show all user info
    try:
        user_row_silver = silver_sheet.find(user_id).row
    except gspread.exceptions.CellNotFound:
        return await ctx.send(f"{member.display_name} does not have any entry in the sheet.")

    basic_info = {
        "Discord Name": get_value(silver_sheet, user_row_silver, DISCORD_NAME),
        "Nation Name": get_value(silver_sheet, user_row_silver, NATION_NAME),
        "Capital": get_value(silver_sheet, user_row_silver, CAPITAL),
        "XP": get_value(silver_sheet, user_row_silver, XP),
        "Silver": get_value(silver_sheet, user_row_silver, SILVER),
        "Tiles": get_value(silver_sheet, user_row_silver, TILES),
        "Culture": get_value(silver_sheet, user_row_silver, CULTURE),
        "Subculture": get_value(silver_sheet, user_row_silver, SUBCULTURE),
        "Religion": get_value(silver_sheet, user_row_silver, RELIGION),
        "Status": get_value(silver_sheet, user_row_silver, STATUS),
        "Title": get_value(silver_sheet, user_row_silver, TITLE),
        "Other Bonuses": get_value(silver_sheet, user_row_silver, OTHER_BONUSES),
    }

    # Attempt to fetch additional data from other sheets
    def try_get_data(sheet, processor):
        try:
            user_row = sheet.find(user_id).row
            return processor(user_row)
        except gspread.exceptions.CellNotFound:
            return {}

    resource_data = try_get_data(resource_sheet, category_map["resources"][1])
    production_data = try_get_data(production_sheet, category_map["production"][1])
    building_data = try_get_data(buildings_sheet, category_map["buildings"][1])
    war_data = try_get_data(war_sheet, category_map["war"][1])

    # Format and send the full embed
    embed = discord.Embed(title=f"{member.display_name}'s Status", color=discord.Color.blue())
    embed.set_thumbnail(url=member.avatar.url if member.avatar else "")

    embed.add_field(name="Basic Info", value="\n".join([f"**{key}:** {value}" for key, value in basic_info.items()]), inline=False)
    embed.add_field(name="Resources", value="\n".join([f"**{key}:** {value}" for key, value in resource_data.items()]), inline=False)
    embed.add_field(name="Production", value="\n".join([f"**{key}:** {value}" for key, value in production_data.items()]), inline=False)
    embed.add_field(name="Buildings", value="\n".join([f"**{key}:** {value}" for key, value in building_data.items()]), inline=False)
    embed.add_field(name="War", value="\n".join([f"**{key}:** {value}" for key, value in war_data.items()]), inline=False)

    await ctx.send(embed=embed)
"""

MAXIMUM_SHORT_LOG_LENGTH = 20
LOG_CHUNK_SIZE = 5
VALID_PARAMS = ["all", "short"]
VALID_ACTIONS = ["latest", "full"]

async def length_autocomplete(
    ctx: Context,
    current: str,
) -> List[app_commands.Choice[str]]:
    return [
        app_commands.Choice(name=param.capitalize(), value=param)
        for param in VALID_PARAMS if current.lower() in param.lower()
    ]

async def action_autocomplete(
    ctx: Context,
    current: str,
) -> List[app_commands.Choice[str]]:
    return [
        app_commands.Choice(name=action.capitalize(), value=action)
        for action in VALID_ACTIONS if current.lower() in action.lower()
    ]

# Config 
CONFIG_PARAMETERS = [
    "expansion_channel",
    "resource_channel",
    "building_channel"
]

async def param_autocomplete(
    ctx: Context,
    current: str,
) -> List[app_commands.Choice[str]]:
    return [
        app_commands.Choice(name=parameter, value=parameter)
        for parameter in CONFIG_PARAMETERS if current.lower() in parameter.lower()
    ]

@bot.hybrid_command(name="config", brief="Set the configuration parameters.")
@app_commands.autocomplete(parameter=param_autocomplete)
async def config_command(ctx: commands.Context, parameter: str):
    """
    !config expansion_channel
    !config resource_channel
    !config building_channel

    Sets the current channel as the specified logging channel.
    """

    if isinstance(ctx.interaction, Interaction) and not ctx.interaction.response.is_done():
        await ctx.interaction.response.defer()  # Acknowledge the interaction immediately

    if not (ctx.author.guild_permissions.administrator or ctx.author.id == 437982451709640706):
        return await ctx.send("You do not have permission to configure the bot. Only Admins are allowed.")


    if parameter.strip().lower()[-8:] == "_channel":
        config[f"{parameter.strip().lower()[:-8]}_channel_id"] = ctx.channel.id
        await ctx.send(f"The {parameter.strip().lower()[:-8]} channel has been set to {ctx.channel.mention}.")
    else:
        await ctx.send("Invalid parameter. Supported parameters: 'expansion_channel', 'building_channel', 'resource_channel'.")

# Temp functions 

#@bot.hybrid_command(name="name", brief="Test command ensuring display and global names.")
async def name(ctx: commands.Context, member: discord.Member):
    """
    @User 
    Example: !name @User
    """
    if isinstance(ctx.interaction, Interaction) and not ctx.interaction.response.is_done():
        await ctx.interaction.response.defer() # Acknowledge the interaction immediately

    if not is_authorized(ctx):
        return await ctx.send("You do not have permission to modify the sheet.")

    user_global_name = member.global_name if member.global_name else member.display_name

    return await ctx.send(f"User display: {member.display_name}; user global: {member.global_name}; user name: {member.name} (chosen: {user_global_name}).")

# Shortcuts 

@bot.hybrid_command(name="trade", brief="Swap resources between two users.")
@app_commands.describe(
    user1="The first user involved in the trade.",
    user1_sends="Resources sent by the first user (e.g., '100 Silver, 2 Crops').",
    user2="The second user involved in the trade.",
    user2_sends="Resources sent by the second user (e.g., '50 Timber, 3 Tools')."
)
async def trade(
    ctx: commands.Context,
    user1: discord.Member,
    user1_sends: str,
    user2: discord.Member,
    user2_sends: str = ""
):
    """
    Trade resources between two users by deducting and adding specified amounts.\n
    ‎\n
    Usage\n
    -----------\n
      !trade @User1 [resources user1 sends] @User2 [resources user2 sends]\n
    ‎\n
    Examples\n
    -----------\n
      - !trade @UserA "100 Silver, 2 Crops" @UserB "30 Timber, 2 Tools"\n
        This means:\n
          - UserA sends 100 Silver and 2 Crops to UserB.\n
          - UserB sends 30 Timber and 2 Tools to UserA.\n
      - !trade @User1 "100 Silver, 2 Crops" @User2 "9 Energy"\n
      - !trade @User1 "5 Silver" @User2\n
      - !trade @User1 "" @User2 "5 Silver"\n

    Parameters
    -----------
    user1: discord.Member
        The first user involved in the trade.
    user1_sends: str
        A comma-separated list of resources and amounts sent by the first user (e.g., '100 Silver, 2 Crops').
    user2: discord.Member
        The second user involved in the trade.
    user2_sends: str, optional
        A comma-separated list of resources and amounts sent by the second user (default is an empty string).
    """
    if isinstance(ctx.interaction, Interaction) and not ctx.interaction.response.is_done():
        await ctx.interaction.response.defer()  # Acknowledge the interaction immediately

    # 1. Authorization check
    if not is_authorized(ctx):
        return await ctx.send("You do not have permission to initiate trades.")

    user1 = get_member(user1)
    user2 = get_member(user2)
    # 2. Parse resource strings
    user1_list, invalid1 = parse_resource_list(user1_sends)
    user2_list, invalid2 = parse_resource_list(user2_sends)

    if invalid1 or invalid2:
        return await ctx.send(f"Invalid resources: {invalid1 + invalid2}. Aborting Trade.")

    if not user1_list and not user2_list:
        return await ctx.send("No resources specified to trade.")

    # 
    user1_changes = []
    for (qty, unit) in user1_list:
        user1_changes.append((-qty, unit))
    for (qty, unit) in user2_list:
        user1_changes.append((+qty, unit))


    user2_changes = []
    for (qty, unit) in user2_list:
        user2_changes.append((-qty, unit))
    for (qty, unit) in user1_list:
        user2_changes.append((+qty, unit))

    can_user1, summary1 = await check_debt(ctx, user1, [(-qty, resource) for (qty, resource) in user1_changes], title="")
    if not can_user1:
        return  # canceled or insufficient resources for user1

    can_user2, summary2 = await check_debt(ctx, user2, [(-qty, resource) for (qty, resource) in user2_changes], title="")
    if not can_user2:
        return  # canceled or insufficient resources for user2

    # Create an embed summarizing the trade
    embed = discord.Embed(title="Trade Summary", color=discord.Color.blue())
    embed.add_field(
        name=f"{user1.display_name} → {user2.display_name}",
        value="\n".join(summary1) if summary1 else "Nothing sent",
        inline=False
    )
    embed.add_field(
        name=f"{user2.display_name} → {user1.display_name}",
        value="\n".join(summary2) if summary2 else "Nothing sent",
        inline=False
    )

    # 3. Apply changes with batch_add_all
    # user1
    result_user1 = batch_add_all(
        ctx=ctx,
        member=user1,
        resource_list=user1_changes,
        log_msg=f"Trade with {user2.name}"
    )
    if not result_user1:
        return await ctx.send("User1 not found. Aborting Trade")

    # user2
    result_user2 = batch_add_all(
        ctx=ctx,
        member=user2,
        resource_list=user2_changes,
        log_msg=f"Trade with {user1.name}"
    )
    if not result_user2:
        return await ctx.send("User2 not found. Trade only registered from on user1. Ask bot creator to intervene.")

    await ctx.send(f"Trade completed between {user1.display_name} and {user2.display_name}.")

def rewrite_sheet(sheet: Worksheet, header_row: List[str], new_records: List[dict]):
    """
    Replaces the entire contents of `sheet` with the given header row + new_records.

    :param sheet: The gspread Worksheet to rewrite
    :param header_row: The row_values(1) result (list of column names)
    :param new_records: a list of dictionaries, each representing a row's data
    """

    # We do a single batch update approach:
    # 1) Clear the sheet, or we can just resize to 1 row
    sheet.clear()

    # 2) Re-append the header
    # Build data: the first row is header_row
    data = [header_row]

    # 3) For each record in new_records, we convert from dict to list matching header
    for rec in new_records:
        row_list = []
        for col_name in header_row:
            row_list.append(str(rec.get(col_name, "")))
        data.append(row_list)

    # 4) do a single update
    # The range can be from A1 downward or we can just use update(data, "A1")
    sheet.update("A1", data, value_input_option="USER_ENTERED")

bot.run(DISCORD_TOKEN)

