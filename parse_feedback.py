# -*- coding: utf-8 -*-
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import telebot
from telebot import types
plt.style.use('dark_background')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import openai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
#nltk.download('omw-1.4')

def get_feedback(root):
    url = f"https://feedbacks1.wb.ru/feedbacks/v1/{root}"
    res = requests.get(url=url)
    temp = json.loads(res.text)
    if temp['feedbacks'] != None:
        return temp
    else:
        for i in range(1,10):
            url = f"https://feedbacks{str(i)}.wb.ru/feedbacks/v1/{root}"
            res = requests.get(url=url)
            temp = json.loads(res.text)
            if temp['feedbacks'] != None:
                return temp

def create_df(feedbacks):
    dfs = []
    for new_row in feedbacks:
         dfs.append(pd.DataFrame([new_row]))
    df = pd.concat(dfs, ignore_index=True)
    return df

def describe(fb_json):
    description = {'Общая оценка': fb_json['valuation'],
    'Продаж': fb_json['valuationSum'],
    'Распределение оценок': fb_json['valuationDistribution'],
    'Размерная сетка':fb_json['matchingSizePercentages'],
    'Отзывов':fb_json['feedbackCount'],
    'Отзывов с фото':fb_json['feedbackCountWithPhoto'],
    'Отзывов с текстом':fb_json['feedbackCountWithText'],
    'Отзывов с видео':fb_json['feedbackCountWithVideo']}
    return description


def preprocess_text(text):
    if not text:  # Check if the text is None or empty
        return ""
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('russian')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def review_clusterization(df):
    reviews_raw = df['text'].tolist()
    reviews = [preprocess_text(review) for review in reviews_raw]
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(reviews)
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    distances = kmeans.transform(X)
    closest_reviews = []
    for i in range(n_clusters):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        cluster_distances = distances[cluster_indices, i]
        sorted_indices = cluster_indices[np.argsort(cluster_distances)]
        n_closest = max(1, len(sorted_indices) // 5)
        closest_reviews.extend(sorted_indices[:n_closest])
    return [reviews[i] for i in closest_reviews]

def chat_gpt_ask(reviews):
    max_tokens = 4096
    temperature = 1
    api_key = 'sk-proj-hxZQ9JwDPT9ahfFz69sGT3BlbkFJZQsbdwAX3QHpsBIDs1vh'
    openai.api_key = api_key
    model = "gpt-3.5-turbo"
    prompt = f"Напиши 3 главные особенности, которые пользователи ценят в товаре, и 3 главные боли, опираясь на отзывы : {reviews}. Будь точен, рузультат верни в виде словаря."
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": 'Ты анализируешь отзывы на товары. Возвращай строго словарь с плюсами и минусами (сгруппируй похожие отзывы и правильно выдели только главные 3 плюса и 3 минуса) товаров по типу {"pluses":["преимущество 1","преимущество 2","преимущество 3"], "minuses":["Минус 1","Минус 2","Минус 3"]}. Ничего больше возвращать нельзя.'},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        res = json.loads(response['choices'][0]['message']['content'].strip())
        return res
    except:
        return json.loads({"error":"Произошла ошибка"})

def create_image(data):
    # Switching to a non-interactive backend
    matplotlib.pyplot.switch_backend('Agg')
    # Создание изображения
    image = Image.open("im.png")
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font_path = "HelveticaMedium.ttf"
    font = ImageFont.truetype(font_path, 52)

    text = f"Общая оценка: {data['Общая оценка']}\nПродаж: {data['Продаж']}\nОтзывов: {data['Отзывов']}\n" \
           f"Отзывов с фото: {data['Отзывов с фото']}\nОтзывов с текстом: {data['Отзывов с текстом']}\n" \
           f"Отзывов с видео: {data['Отзывов с видео']}"
    draw.text((50, 50), text, fill='white', font=font)
    ratings = list(data['Распределение оценок'].keys())
    counts = list(data['Распределение оценок'].values())
    plt.figure(figsize=(8, 6))
    sns.set_style("dark")
    sns.barplot(x=ratings, y=counts, color='#4939b5')  # Adjust the color here
    plt.title('Распределение оценок', color='white')
    plt.xlabel('Оценка', color='white')
    plt.ylabel('Количество', color='white')
    plt.gca().set_facecolor('black')
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.3)

    # Сохранение графика в отдельное изображение
    chart_image_path = "chart_image.png"
    plt.savefig(chart_image_path)
    image_path = "final_image.png"
    image.save(image_path)
    plt.close()
    return image_path, chart_image_path

def add_image_to_corner(photo1_path, photo2_path, output_path):
    photo1 = Image.open(photo1_path)
    photo2 = Image.open(photo2_path)
    width1, height1 = photo1.size
    width2, height2 = photo2.size
    x = width1 - width2
    y = 0
    photo1.paste(photo2, (x, y))
    photo1.save(output_path)
    photo1.close()
    photo2.close()
    return output_path

def add_text(data, photo_path):
    image = Image.open(photo_path)
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font_path = "HelveticaMedium.ttf"
    font = ImageFont.truetype(font_path, 52)

    pluses = data['pluses']
    minuses = data['minuses']

    text = "Плюсы:\n" + "\n".join(f"- {plus}" for plus in pluses)
    text += "\n\nМинусы:\n" + "\n".join(f"- {minus}" for minus in minuses)

    text_position = (50, image.height - 500)
    draw.text(text_position, text, fill='white', font=font)
    output_path = photo_path
    image.save(output_path)
    image.close()
    return output_path, text



if __name__ == "__main__":
    TOKEN = '7030288634:AAEDXQjUAYhBCBmtFGx2UVbafT3bF3VT4m4'
    openai.api_key = 'sk-proj-hxZQ9JwDPT9ahfFz69sGT3BlbkFJZQsbdwAX3QHpsBIDs1vh'
    bot = telebot.TeleBot(TOKEN)

    @bot.message_handler(commands=['start'])
    def url(message):
        markup = types.InlineKeyboardMarkup()
        btn1 = types.InlineKeyboardButton(text='По любым вопросам', url='https://t.me/ogandr')
        markup.add(btn1)
        bot.send_message(message.from_user.id,
                         "Привет!\n\nЯ Feedy, главный по аналитике отзывов 👁👁\nЯ помогу тебе получить много полезной информации из отзывов на твоих товарах или у конкурента\nСвязаться можно по кнопке ниже",
                         reply_markup=markup)
        bot.send_message(message.from_user.id, "Отправь айдишник товара, будем смотреть 👁👁")

    @bot.message_handler(content_types=['text'])
    def get_text_messages(message):
        try:
            id = int(message.text)
            bot.send_message(message.from_user.id, "Обрабатываю... Пару сек")
            fb_json = get_feedback(id)
            df = create_df(fb_json['feedbacks'])
            ids = list(df['id'])
            print(ids)
            df.to_csv('temp.csv')
            description = describe(fb_json)
            print(description)
            image_parts = create_image(description) #фотка с описанием
            print('ok')
            image = add_image_to_corner(image_parts[0], image_parts[1], 'final_image.png')
            print('ok2')
            list_of_reviews = review_clusterization(df)
            print(list_of_reviews)
            advanced_review = chat_gpt_ask(list_of_reviews)
            print(advanced_review)
            image, text = add_text(advanced_review, 'final_image.png')
            #text_with_review = make_text(advanced_review)
            # bot.send_message(message.from_user.id, advanced_review)
            bot.send_photo(message.from_user.id, photo=open(image, 'rb'), caption=text)
            #bot.send_document(message.from_user.id, document='temp.csv', caption=advanced_review)
        except:
            bot.send_message(message.from_user.id, "Чет не то, попробуй другой айдишник 👁👁")
    bot.polling(none_stop=True, interval=0)


