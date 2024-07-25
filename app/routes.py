# from flask import Blueprint, render_template, current_app
# import os

# main = Blueprint('main', __name__)

# @main.route('/')
# def index():
#     video_folder = current_app.config['VIDEO_FOLDER']
#     if not os.path.exists(video_folder):
#         os.makedirs(video_folder)

#     videos = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
#     if not videos:
#         return "No videos found in the directory."
#     return render_template('index.html', videos=videos)