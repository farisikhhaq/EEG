from flask import render_template, request, session, redirect, flash, abort
from app.models.user import User

class AuthController:
    def login_view():
        return render_template('auth/login.html')
    
    def login_process():
        email, password = request.form['email'], request.form['password']
    
        user = User.query.filter_by(email=email).first()
        if(user and user.verify_password(password)):
            session['user_id'] = user.id
            session['username'] = user.username
            session['email'] = user.email
            session['role'] = user.role
            return redirect('/dashboard')
        else:
            flash("The credentials doesn't match our records!", "message")
            return redirect(request.referrer or '/')
        
    def logout():
        session.pop('user_id', None)
        session.pop('username', None)
        session.pop('email', None)
        session.pop('role', None)
        flash("You’ve been logged out.", "message")
        return redirect('/')
    
    def global_middleware():
        if request.endpoint in ('login', 'static'):
            return
        
        if request.path == '/login':
            if session.get('user_id'):
                return redirect('/dashboard')
        else:
            if not session.get('user_id'):
                return redirect('/login')    

        protected_routes = {
            '/user': ['super_admin']
        }

        allowed_roles = protected_routes.get(request.path)
        if allowed_roles:
            user_role = session.get('role')
            if not user_role or user_role not in allowed_roles:
                return abort(403)