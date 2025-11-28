from app import db
from datetime import datetime
from zoneinfo import ZoneInfo
import enum

class TypeEnum(str, enum.Enum):
    training = 'training'
    testing = 'testing'

class ModelTypeEnum(str, enum.Enum):
    nb = 'nb'
    svm = 'svm'
    rf = 'rf'

class ExtractionTypeEnum(str, enum.Enum):
    time = 'time'
    freq = 'freq'
    both = 'both'

class Log(db.Model):
    __tablename__ = 'logs'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    type = db.Column(db.Enum(TypeEnum), nullable=False)
    model_type = db.Column(db.Enum(ModelTypeEnum), nullable=False)
    extraction_type = db.Column(db.Enum(ExtractionTypeEnum), nullable=False)
    model_path = db.Column(db.String(255), nullable=True)
    scaler_path = db.Column(db.String(255), nullable=True)
    accuracy = db.Column(db.String(7), nullable=True)
    execution_time = db.Column(db.Integer, nullable=True)
    classification_report = db.Column(db.JSON, nullable=True)
    confusion_matrix = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(ZoneInfo("Asia/Jakarta")))