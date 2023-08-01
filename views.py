from website.static.logistics import process_video
from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from .models import Note
from . import db
import json
from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired

views = Blueprint('views', __name__)

upload_folder = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),upload_folder,secure_filename(file.filename))) # Then save the file
        print(file.filename)
        val = "c:\\Users\\sinen\\Desktop\\Dance-Q\\website\\static\\files\\" + file.filename
        print(val)
        val = process_video(val)
        data_is = "For: " + file.filename + ", the sync quotient is: " + str(val)
        new_note = Note(data=data_is, user_id=current_user.id)
        db.session.add(new_note)
        db.session.commit()
        flash('Note added!', category='success')

    return render_template("home.html", user=current_user, form=form)


@views.route('/delete-note', methods=['POST'])
def delete_note():
    note = json.loads(request.data)
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()

    return jsonify({})
