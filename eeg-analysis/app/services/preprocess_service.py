import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import iirnotch, butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.signal import welch
import zipfile
import io
import time
import tempfile
from app.models.log import Log, TypeEnum, ModelTypeEnum, ExtractionTypeEnum
import joblib
import os
import uuid
from flask import session
from datetime import datetime
from zoneinfo import ZoneInfo
from app import db

class PreprocessService:
    def process(uploaded_file, model_type='nb', extraction_mode='both'):
        start_time = time.time()
        result = PreprocessService.load_and_process_data(uploaded_file=uploaded_file, model_type=model_type, extraction_mode=extraction_mode)
        end_time = time.time()
        execution_time = end_time - start_time

        print(execution_time)

        if result['success']:
            new_experiment = Log(
                user_id = session['user_id'],
                type = TypeEnum.training,
                model_type = ModelTypeEnum(model_type),
                extraction_type = ExtractionTypeEnum(extraction_mode),
                scaler_path = result['scaler_path'],
                model_path = result['model_path'],
                accuracy = result['accuracy'],
                execution_time = int(execution_time),
                classification_report = result['classification_report'],
                confusion_matrix = result['confusion_matrix'],
                created_at=datetime.now(ZoneInfo("Asia/Jakarta"))
            )
            db.session.add(new_experiment)
            db.session.commit()

        return result
    
    def apply_notch_filter(data, f0, fs, quality_factor):
        nyquist = 0.5 * fs
        freq = f0 / nyquist
        b, a = iirnotch(freq, quality_factor)
        return filtfilt(b, a, data)
    
    def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def apply_windowing(data, window_size, step_size):
        return np.array([data[i:i+window_size] for i in range(0, len(data) - window_size + 1, step_size)])
    
    def extract_features(windows, n_channels=5, fs=250, mode='both'):
        """
        windows: shape (n_windows, window_length * n_channels)
        mode: 'time', 'freq', atau 'both'
        """
        window_len = windows.shape[1] // n_channels
        features = []

        for window in windows:
            window = window.reshape(window_len, n_channels)
            feats = []

            for ch in range(n_channels):
                signal = window[:, ch]

                if mode in ['time', 'both']:
                    feats.extend([
                        np.mean(signal),
                        np.std(signal),
                        np.min(signal),
                        np.max(signal)
                    ])

                if mode in ['freq', 'both']:
                    freqs, psd = welch(signal, fs, nperseg=fs)
                    bands = {
                        'delta': (0.5, 4),
                        'theta': (4, 8),
                        'alpha': (8, 12),
                        'beta': (12, 30),
                        'gamma': (30, 45),
                    }
                    for band in bands.values():
                        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
                        power = np.trapz(psd[idx], freqs[idx])
                        feats.append(power)

            features.append(feats)

        return np.array(features)

    def process_file(file_path, fs, window_size, step_size, label, mode):
        df = pd.read_csv(file_path, skiprows=4)
        df.columns = df.columns.str.strip()
        features_all_channels = []
        
        signal_columns = df.select_dtypes(include='number').columns
        
        for col in signal_columns:
            signal = df[col].values
            signal = PreprocessService.apply_notch_filter(signal, f0=50, fs=fs, quality_factor=30)
            signal = PreprocessService.apply_bandpass_filter(signal, lowcut=0.5, highcut=40, fs=fs)
            windows = PreprocessService.apply_windowing(signal, window_size, step_size)
            features = PreprocessService.extract_features(windows, mode=mode)  # shape: (num_windows, 4)
            features_all_channels.append(features)
        
        combined_features = np.concatenate(features_all_channels, axis=1)  # (num_windows, num_channels*4)
        labels = [label] * combined_features.shape[0]
        
        return combined_features, labels
    
    def load_and_process_data(uploaded_file, fs=250, window_size=250, step_size=125, extraction_mode="both", model_type="nb"):
        X_all = []
        y_all = []

        category_map = {
            'viat-map': 0,
            'reading': 1,
            'relax': 2,
            'go-nogo': 3,
        }

        zip_data = zipfile.ZipFile(io.BytesIO(uploaded_file.read()))
        temp_files = []

        try:
            for file_info in zip_data.infolist():
                if file_info.filename.endswith('.txt') and not file_info.is_dir():
                    for key, label in category_map.items():
                        if key in file_info.filename:
                            with zip_data.open(file_info) as file:
                                with tempfile.NamedTemporaryFile(mode='w+b', suffix='.txt', delete=False) as tmp:
                                    tmp.write(file.read())
                                    tmp.flush()
                                    temp_files.append(tmp.name)
                                    X, y = PreprocessService.process_file(
                                        tmp.name,
                                        fs=fs,
                                        window_size=window_size,
                                        step_size=step_size,
                                        label=label,
                                        mode=extraction_mode
                                    )
                                    X_all.append(X)
                                    y_all.extend(y)
                            break
        finally:
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
        result = PreprocessService.train(X=np.vstack(X_all), y=y_all, model_type=model_type)

        return result


    
    def train(X, y, model_type):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



        if model_type == 'svm':
            model = SVC(kernel='rbf', C=1.0, gamma='scale')
        elif model_type == 'nb':
            model = GaussianNB()
        elif model_type == 'rf':
            model = RandomForestClassifier()
        else:
            return {
                "success": False, 
                "message": "Model not found!",
            }
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        scaler_path, model_path = PreprocessService.save_file(model, scaler, model_type=model_type)

        category_map = {
            'viat-map': 0,
            'reading': 1,
            'relax': 2,
            'go-nogo': 3,
        }

        labels = list(category_map.values()) 
        class_names = list(category_map.keys()) 

        cm = confusion_matrix(y_test, y_pred, labels=labels)

        return {
            "success": True,
            "message": "Training success! Please check the logs for details!",
            "classification_report": classification_report(y_test, y_pred, target_names=['viat-map', 'reading', 'relax', 'go-nogo'], output_dict=True),
            "accuracy": f"{accuracy_score(y_test, y_pred) * 100:.2f}",
            "confusion_matrix": {
                'matrix': cm.tolist(),
                "classes": class_names
            },
            "scaler_path": scaler_path,
            "model_path": model_path
        }
    
    def save_file(model, scaler, model_type):
        unique_id = str(uuid.uuid4())
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.abspath(os.path.join(base_dir, '../../static/data/training'))
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, f'{unique_id}_{model_type}_model.pkl')
        scaler_path = os.path.join(output_dir, f'{unique_id}_{model_type}_scaler.pkl')

        try:
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
        except Exception as e:
            print("❌ Gagal menyimpan file:", e)
            raise

        return scaler_path, model_path
    
    def load_and_test_data(uploaded_file, uploaded_model, uploaded_scaler, fs=250, window_size=250, step_size=125, extraction_mode="both"):
        try:
            X_all = []
            y_all = []
            temp_files = []

            category_map = {
                'viat-map': 0,
                'reading': 1,
                'relax': 2,
                'go-nogo': 3,
            }

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_model:
                tmp_model.write(uploaded_model.read())
                tmp_model.flush()
                temp_files.append(tmp_model.name)
                model = joblib.load(tmp_model.name)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_scaler:
                tmp_scaler.write(uploaded_scaler.read())
                tmp_scaler.flush()
                temp_files.append(tmp_scaler.name)
                scaler = joblib.load(tmp_scaler.name)


            zip_data = zipfile.ZipFile(io.BytesIO(uploaded_file.read()))

            for file_info in zip_data.infolist():
                if file_info.filename.endswith('.txt') and not file_info.is_dir():
                    for key, label in category_map.items():
                        if key in file_info.filename:
                            with zip_data.open(file_info) as file:
                                with tempfile.NamedTemporaryFile(mode='w+b', suffix='.txt', delete=False) as tmp:
                                    tmp.write(file.read())
                                    tmp.flush()
                                    temp_files.append(tmp.name)
                                    X, y = PreprocessService.process_file(
                                        tmp.name,
                                        fs=fs,
                                        window_size=window_size,
                                        step_size=step_size,
                                        label=label,
                                        mode=extraction_mode
                                    )
                                    X_all.append(X)
                                    y_all.extend(y)
                            break

            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

            result = PreprocessService.test(X=np.vstack(X_all), y=y_all, model=model, scaler=scaler)

            return result

        except Exception as e:
            print(str(e))
            return {
                "success": False,
                "message": f"Terjadi kesalahan saat testing: {str(e)}"
            }

    def test(X, y, model, scaler):
        X = scaler.fit_transform(X)

        y_pred = model.predict(X)

        category_map = {
            'viat-map': 0,
            'reading': 1,
            'relax': 2,
            'go-nogo': 3,
        }

        labels = list(category_map.values()) 
        class_names = list(category_map.keys()) 

        cm = confusion_matrix(y, y_pred, labels=labels)

        type_model = {
            "GaussianNB": 'nb',
            "SVC": 'svm',
            'RandomForestClassifier': 'rf'
        }

        return {
            "success": True,
            "message": "Training success! Please check the logs for details!",
            "classification_report": classification_report(y, y_pred, target_names=['viat-map', 'reading', 'relax', 'go-nogo'], output_dict=True),
            "accuracy": f"{accuracy_score(y, y_pred) * 100:.2f}",
            "model_type": type_model[type(model).__name__],
            "confusion_matrix": {
                'matrix': cm.tolist(),
                "classes": class_names
            }
        }


    def test_model(uploaded_file, uploaded_model, uploaded_scaler, extraction_mode="both"):
        start_time = time.time()
        result = PreprocessService.load_and_test_data(uploaded_file=uploaded_file, uploaded_model=uploaded_model, uploaded_scaler=uploaded_scaler, extraction_mode=extraction_mode)
        end_time = time.time()

        execution_time = end_time - start_time

        if result['success']:
            new_test = Log(
                user_id = session['user_id'],
                type = TypeEnum.testing,
                model_type = ModelTypeEnum(result['model_type']),
                extraction_type = ExtractionTypeEnum(extraction_mode),
                accuracy = result['accuracy'],
                execution_time = int(execution_time),
                classification_report = result['classification_report'],
                confusion_matrix = result['confusion_matrix'],
                created_at=datetime.now(ZoneInfo("Asia/Jakarta"))
            )
            db.session.add(new_test)
            db.session.commit()

        return result

