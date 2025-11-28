from flask import render_template, request

class DashboardController:
    def index():
        return render_template('dashboard/index.html', title="Dashboard", current_url=request.url)
        