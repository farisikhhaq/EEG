from flask import render_template, request, flash, redirect, send_file
from app.services.preprocess_service import PreprocessService
from app.models.log import Log
import os
import io
import csv

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

    def download_report(id):
        """Download log report as CSV or TXT"""
        fmt = request.args.get('format', 'csv')  # 'csv' or 'txt'
        log = Log.query.get_or_404(id)

        # Build report content
        cr = log.classification_report or {}
        cm = log.confusion_matrix or {}

        # Mapping for display names
        type_map = {'training': 'Training', 'testing': 'Testing'}
        model_map = {'nb': 'Naive Bayes', 'svm': 'SVM', 'rf': 'Random Forest'}
        extraction_map = {'time': 'Time Domain', 'freq': 'Freq Domain', 'both': 'Time & Freq Domain'}

        log_type = type_map.get(str(log.type.value) if hasattr(log.type, 'value') else str(log.type), str(log.type))
        log_model = model_map.get(str(log.model_type.value) if hasattr(log.model_type, 'value') else str(log.model_type), str(log.model_type))
        log_extraction = extraction_map.get(str(log.extraction_type.value) if hasattr(log.extraction_type, 'value') else str(log.extraction_type), str(log.extraction_type))

        if fmt == 'csv':
            output = io.StringIO()
            writer = csv.writer(output)

            # Metadata section
            writer.writerow(['EEG Analysis Report'])
            writer.writerow([])
            writer.writerow(['Type', log_type])
            writer.writerow(['Model', log_model])
            writer.writerow(['Extraction', log_extraction])
            writer.writerow(['Accuracy', f"{log.accuracy}%"])
            writer.writerow(['Execution Time', f"{log.execution_time}s"])
            writer.writerow(['Created At', str(log.created_at)])
            writer.writerow([])

            # Classification Report
            writer.writerow(['Classification Report'])
            writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
            for label, metrics in cr.items():
                if label != 'accuracy' and isinstance(metrics, dict):
                    writer.writerow([
                        label,
                        f"{metrics.get('precision', 0):.4f}",
                        f"{metrics.get('recall', 0):.4f}",
                        f"{metrics.get('f1-score', 0):.4f}",
                        metrics.get('support', '')
                    ])
            writer.writerow([])

            # Confusion Matrix
            if cm:
                writer.writerow(['Confusion Matrix'])
                classes = cm.get('classes', [])
                matrix = cm.get('matrix', [])
                writer.writerow(['Actual \\ Predicted'] + classes)
                for i, row in enumerate(matrix):
                    class_name = classes[i] if i < len(classes) else f'Class {i}'
                    writer.writerow([class_name] + row)

            content = output.getvalue()
            output.close()

            buf = io.BytesIO(content.encode('utf-8'))
            filename = f"eeg_report_{log.id}_{log_type.lower()}.csv"
            return send_file(buf, as_attachment=True, download_name=filename, mimetype='text/csv')

        else:  # TXT format
            lines = []
            lines.append('=' * 60)
            lines.append('EEG ANALYSIS REPORT')
            lines.append('=' * 60)
            lines.append('')
            lines.append(f'Type           : {log_type}')
            lines.append(f'Model          : {log_model}')
            lines.append(f'Extraction     : {log_extraction}')
            lines.append(f'Accuracy       : {log.accuracy}%')
            lines.append(f'Execution Time : {log.execution_time}s')
            lines.append(f'Created At     : {log.created_at}')
            lines.append('')
            lines.append('-' * 60)
            lines.append('CLASSIFICATION REPORT')
            lines.append('-' * 60)
            lines.append(f'{"Class":<20} {"Precision":<12} {"Recall":<12} {"F1-Score":<12} {"Support":<10}')
            lines.append('-' * 66)
            for label, metrics in cr.items():
                if label != 'accuracy' and isinstance(metrics, dict):
                    lines.append(
                        f'{label:<20} '
                        f'{metrics.get("precision", 0):<12.4f} '
                        f'{metrics.get("recall", 0):<12.4f} '
                        f'{metrics.get("f1-score", 0):<12.4f} '
                        f'{str(metrics.get("support", "")):<10}'
                    )
            lines.append('')

            # Confusion Matrix
            if cm:
                lines.append('-' * 60)
                lines.append('CONFUSION MATRIX')
                lines.append('-' * 60)
                classes = cm.get('classes', [])
                matrix = cm.get('matrix', [])
                header = f'{"Actual / Predicted":<20}' + ''.join(f'{c:<12}' for c in classes)
                lines.append(header)
                lines.append('-' * len(header))
                for i, row in enumerate(matrix):
                    class_name = classes[i] if i < len(classes) else f'Class {i}'
                    row_str = f'{class_name:<20}' + ''.join(f'{str(v):<12}' for v in row)
                    lines.append(row_str)

            lines.append('')
            lines.append('=' * 60)

            content = '\n'.join(lines)
            buf = io.BytesIO(content.encode('utf-8'))
            filename = f"eeg_report_{log.id}_{log_type.lower()}.txt"
            return send_file(buf, as_attachment=True, download_name=filename, mimetype='text/plain')
