from flask import render_template, request, session, flash, redirect
from app import db
from app.models.user import User
from sqlalchemy import or_

class UserController:
    def render_user():
        users = User.query.filter(User.id != session.get('user_id')).all()
        return render_template('user/index.html', users=users, current_url=request.url, title="User Management")
    
    def update_or_delete_user(id):
        method = request.form.get('_method', '').upper()
        if method == "PUT":
            username = request.form['username']
            email = request.form['email']
            password = request.form.get('password')
            role = request.form['role']

            user = User.query.get_or_404(id)
            user.username = username
            user.email = email
            user.role = role
            if password:
                user.password = password
            db.session.commit()

            flash("User updated!", "message")
            flash(True, "success")
            return redirect(request.referrer or '/dashboard')
        elif method == "DELETE":
            user = User.query.get_or_404(id)
            db.session.delete(user)
            db.session.commit()
            flash("User deleted!", "message")
            flash(True, "success")
            return redirect(request.referrer or '/dashboard')
        flash("Method not allowed!", "message")
        flash(False, "success")
        return redirect(request.referrer or '/dashboard')
    
    def create_user():
        username, email, password, role = request.form['username'], request.form['email'], request.form['password'], request.form.get('role')

        if(not username or not email or not password):
            flash("Fill all of those credentials!", "message")
            flash(False, "success")
            return redirect(request.referrer or '/')
        
        prev_user = User.query.filter(or_(User.email == email, User.username == username)).first()
        if(prev_user):
            flash("The username or email already taken!", "message")
            flash(False, "success")
            return redirect(request.referrer or '/')
        
        user = User(
            email=email,
            username=username,
            role=role
        )
        user.password = password
        db.session.add(user)
        db.session.commit()

        flash("User created!", "message")
        flash(True, "success")
        return redirect(request.referrer or '/')