from flask.cli import with_appcontext
import click
from seeders.user_seeder import seed_users
from app import db 

@click.command(name='seed_db')
@with_appcontext
def seed_command():
    """Command line interface untuk seeding database"""
    try:
        db.create_all()
        
        seed_users()
        
        db.session.commit()
        print("✅ All seeding completed successfully!")
    except Exception as e:
        db.session.rollback()
        print(f"❌ Seeding failed: {str(e)}")
        raise e