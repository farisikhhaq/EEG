from flask import render_template, request, flash, redirect, session
from app.models.log import Log
from app import db
import os

class LogsController:
    def index():
        page = request.args.get('page', 1, type=int)
        per_page = 10

        mapping = {
            'training': 'Training',
            'testing': 'Testing',
            'nb': 'Naive Bayes',
            'svm': 'SVM',
            'rf': 'Random Forest',
            'time': 'Time Domain',
            'freq': 'Freq Domain',
            'both': 'Time & Freq Domain'
        }

        logs = Log.query.filter_by(user_id=session['user_id']).order_by(Log.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
        for log in logs:
            log.model_type = mapping[log.model_type]
            log.extraction_type = mapping[log.extraction_type]
            log.type = mapping[log.type]

        return render_template('logs/index.html', title="Logs", current_url=request.url, logs=logs)
    
    def delete(id):
        try:
            log = Log.query.get_or_404(id)

            if os.path.exists(log.model_path):
                os.remove(log.model_path)
            
            if os.path.exists(log.scaler_path):
                os.remove(log.scaler_path)

            db.session.delete(log)
            db.session.commit()

            flash(True, "success")
            flash("Delete data successfully!", "message")
            return redirect(request.referrer or '/')
        except Exception as e:
            db.session.rollback()
            flash(False, "success")
            flash("Error while deleting data!", "message")
            return redirect(request.referrer or '/')