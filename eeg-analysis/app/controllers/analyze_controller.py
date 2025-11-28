from flask import render_template, request, flash, redirect, send_file
from app.services.preprocess_service import PreprocessService
import os

class AnalyzeController:
    def testing_view():
        return render_template('analyze/testing.html', title="Testing", current_url=request.url)
    
    def training_view():
        return render_template('analyze/training.html', title="Training", current_url=request.url)
    
    def testing_process():
        dataset_file = request.files.get('dataset')
        model_file = request.files.get('model')
        scaler_file = request.files.get('scaler')
        extraction_mode = request.form.get('extraction')

        if not dataset_file or not model_file or not scaler_file or not extraction_mode:
            flash(False, "success")
            flash("File or request not found!", "message")
            return redirect(request.referrer or '/')

        result = PreprocessService.test_model(uploaded_file=dataset_file, uploaded_model=model_file, uploaded_scaler=scaler_file, extraction_mode=extraction_mode)

        flash(result['success'], "success")
        flash(result['message'], "message")
        return redirect(request.referrer or '/')
    
    def training_process():
        dataset, model, extraction_mode = request.files.get('dataset'), request.form.get('model'), request.form.get('extraction')

        result = PreprocessService.process(uploaded_file=dataset, model_type=model, extraction_mode=extraction_mode)

        flash(result['success'], "success")
        flash(result['message'], "message")
        return redirect(request.referrer or '/')
    
    def download():
        file_path = request.form.get('file_name') 
        if not os.path.isfile(file_path):
            flash(False, "success")
            flash("File not found!", "message")
            return redirect(request.referrer or '/')

        file_name = os.path.basename(file_path)
        return send_file(file_path, as_attachment=True, download_name=file_name)
    
