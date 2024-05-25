import os
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from io import BytesIO

# Your bot token
TOKEN = '6578344747:AAE0odwFwcaFToHJC6_33vUaHSa_LxCbyyc'

# URL of your FastAPI application
API_URL = 'http://127.0.0.1:8000/predict'
API_URL_MULTIPLE = 'http://127.0.0.1:8000/predict_multiple'

# Translation dictionary
fruit_translation = {
    'apple': 'խնձոր',
    'cabbage': 'կաղամբ',
    'carrot': 'գազար',
    'cucumber': 'վարունգ',
    'eggplant': 'սմբուկ',
    'pear': 'տանձ',
    'zucchini': 'դդմիկ'
}


async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Ողջույն! Դու եկել ես մեր մրգերի կանխատեսման մոդելի բոտը։ Խնդրում եմ մեկ ժամվա ընթացքում 10նկարից ավել չուղարկել(Let's not torture Edvard's computer)։")


async def help_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Ինձ ուղարկիր միայն նկարը։')


async def handle_photo(update: Update, context: CallbackContext) -> None:
    try:
        photo = await update.message.photo[-1].get_file()
        photo_path = photo.file_path

        response = requests.get(photo_path)
        image = BytesIO(response.content)

        files = {'file': image}
        response = requests.post(API_URL, files=files, timeout=60)

        if response.status_code == 200:
            prediction = response.json().get('prediction')
            translated_prediction = fruit_translation.get(prediction, prediction)
            await update.message.reply_text(f'Պտի որ {translated_prediction} ըլնի:')
        else:
            await update.message.reply_text('Ցավոք, այս ֆորմատի նկարների հետ պրոցես չի լինում։')
    except Exception as e:
        await update.message.reply_text(f'An error occurred: {e}')


async def handle_multiple_photos(update: Update, context: CallbackContext) -> None:
    try:
        photos = update.message.photo
        image_files = []

        for photo in photos:
            file = await photo.get_file()
            file_path = file.file_path
            response = requests.get(file_path)
            image = BytesIO(response.content)
            image_files.append(('files', ('image.jpg', image, 'image/jpeg')))

        response = requests.post(API_URL_MULTIPLE, files=image_files, timeout=60)

        if response.status_code == 200:
            predictions = response.json().get('predictions')
            translated_predictions = [fruit_translation.get(pred, pred) for pred in predictions]
            predictions_str = "\n".join(translated_predictions)
            await update.message.reply_text(f'Պտի որ \n{translated_predictions}  ընլնեն:')
        else:
            await update.message.reply_text('Ցավոք, այս ֆորմատի նկարների հետ պրոցես չի լինում։')
    except Exception as e:
        await update.message.reply_text(f'An error occurred: {e}')


def main() -> None:
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_photo))

    application.run_polling()


if __name__ == '__main__':
    main()