import os
import tempfile
import time
import shutil
import asyncio
import sys
import pathlib
import html

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)
from telegram.request import HTTPXRequest

# ---------------- CONFIG ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN")

MAX_FILE_MB = 8
USER_COOLDOWN = 60
SESSION_EXPIRY = 600
MAX_FLAGS = 6
MAX_MSG_LEN = 3800

sessions = {}
last_used = {}

# ---------- START ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🧠 *Forensight Bot*\n"
        "Developed by *Spider Anongreyhat*\n\n"
        "Upload an image to analyze.\n"
        "Results are displayed directly.\n"
        f"Max file size: {MAX_FILE_MB}MB",
        parse_mode="Markdown"
    )

# ---------- IMAGE ----------
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    now = time.time()

    if user_id in last_used and now - last_used[user_id] < USER_COOLDOWN:
        await update.message.reply_text("⏳ Please wait before starting another analysis.")
        return

    photo = update.message.photo[-1]
    file = await photo.get_file()

    if file.file_size and file.file_size > MAX_FILE_MB * 1024 * 1024:
        await update.message.reply_text("❌ File too large.")
        return

    temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="forensight_"))
    img_path = temp_dir / "input.jpg"

    await file.download_to_drive(str(img_path))

    sessions[user_id] = {
        "image": img_path,
        "flags": set(),
        "dir": temp_dir
    }

    asyncio.create_task(expire_session(user_id, SESSION_EXPIRY))

    await update.message.reply_text(
        "Select analysis options:",
        reply_markup=keyboard(user_id)
    )

# ---------- KEYBOARD ----------
def keyboard(user_id):
    flags = sessions[user_id]["flags"]

    def mark(x): 
        return "☑" if x in flags else "☐"

    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"{mark('deep-search')} Deep Search", callback_data="deep-search")],
        [InlineKeyboardButton(f"{mark('restore-exif')} Restore EXIF", callback_data="restore-exif")],
        [InlineKeyboardButton(f"{mark('face-count')} Face Count", callback_data="face-count")],
        [InlineKeyboardButton(f"{mark('scene-loc')} Scene Location Guess", callback_data="scene-loc")],
        [InlineKeyboardButton(f"{mark('offline')} Offline Mode", callback_data="offline")],
        [InlineKeyboardButton(f"{mark('queries-only')} Queries Only", callback_data="queries-only")],
        [InlineKeyboardButton("▶ Run Analysis", callback_data="run")]
    ])

# ---------- TOGGLE ----------
async def toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id

    if uid not in sessions:
        await query.edit_message_text("❌ Session expired.")
        return

    data = query.data
    flags = sessions[uid]["flags"]

    if data == "run":
        await query.edit_message_text("🔍 Analysis running…")
        asyncio.create_task(run_analysis(query, uid))
        return

    if data in flags:
        flags.remove(data)
    else:
        if len(flags) >= MAX_FLAGS:
            await query.answer("Too many options selected", show_alert=True)
            return
        flags.add(data)

    await query.edit_message_reply_markup(reply_markup=keyboard(uid))

# ---------- RUN ANALYSIS ----------
async def run_analysis(query, uid):
    session = sessions.get(uid)
    if not session:
        return

    img = session["image"]

    cmd = [
        sys.executable,
        "main.py",
        "--image", str(img)
    ]

    for f in session["flags"]:
        cmd.append(f"--{f}")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )

    stdout, _ = await proc.communicate()
    text = stdout.decode(errors="ignore").strip()

    if not text:
        text = "⚠️ No analysis output."

    safe_text = html.escape(text)

    for i in range(0, len(safe_text), MAX_MSG_LEN):
        await query.message.reply_text(
            safe_text[i:i + MAX_MSG_LEN],
            parse_mode="HTML",
            disable_web_page_preview=True
        )

    cleanup(uid)

# ---------- CLEAN ----------
def cleanup(uid):
    last_used[uid] = time.time()
    if uid in sessions:
        shutil.rmtree(sessions[uid]["dir"], ignore_errors=True)
        sessions.pop(uid, None)

async def expire_session(uid, delay):
    await asyncio.sleep(delay)
    cleanup(uid)

# ---------- MAIN ----------
def main():
    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .request(HTTPXRequest())
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(CallbackQueryHandler(toggle))

    print("✅ Forensight Telegram Bot running…")
    app.run_polling()

if __name__ == "__main__":
    main()
