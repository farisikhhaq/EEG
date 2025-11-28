from flask import Blueprint, redirect
from app.controllers.auth_controller import AuthController
from app.controllers.dashboard_controller import DashboardController
from app.controllers.analyze_controller import AnalyzeController
from app.controllers.logs_controller import LogsController
from app.controllers.user_controller import UserController

web = Blueprint('web', __name__)

@web.before_request
def global_middleware():
    return AuthController.global_middleware()

@web.get('/')
def index():
    return redirect('/dashboard')

@web.get('/login')
def login_view():
    return AuthController.login_view()

@web.post('/login')
def login_process():
    return AuthController.login_process()

@web.get('/user')
def render_user():
    return UserController.render_user()

@web.post('/user/<int:id>')
def update_or_delete_user(id):
    return UserController.update_or_delete_user(id)

@web.post('/user')
def create_user():
    return UserController.create_user()

@web.post('/logout')
def logout():
    return AuthController.logout()

@web.get('/dashboard')
def dashboard_view():
    return DashboardController.index()

@web.get('/training')
def training_view():
    return AnalyzeController.training_view()

@web.post('/training')
def training_process():
    return AnalyzeController.training_process()

@web.get('/testing')
def analyze_view():
    return AnalyzeController.testing_view()

@web.post('/testing')
def testing_process():
    return AnalyzeController.testing_process()

@web.get('/logs')
def logs_view():
    return LogsController.index()

@web.post('/logs/delete/<id>')
def delete_logs(id):
    return LogsController.delete(id)

@web.post('/download')
def download():
    return AnalyzeController.download()