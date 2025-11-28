from app.models.user import User
from werkzeug.security import generate_password_hash
from app import db

def seed_users():
    if not User.query.filter_by(email="super_admin@gmail.com").first():
        super_admin = User(
            username="Super Admin",
            email="super_admin@gmail.com",
            role="super_admin"
        )
        super_admin.password = "12345678"
        db.session.add(super_admin)

    print("User seeding done!")