import os
import uuid
from functools import wraps
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from pytz import timezone
from sqlalchemy import func

from ml_risk_model import ProjectRiskModel
MAX_CAMPAIGN_DAYS = 90
# Создаём экземпляр модели. При первом запуске обучится, затем будет загружать best_model.pkl
risk_model = ProjectRiskModel(force_retrain=False)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crowdfunding.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Настройки загрузки файлов
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = SQLAlchemy(app)

# ========== ФИЛЬТР ДЛЯ ВРЕМЕНИ ==========
def to_moscow_time(dt):
    msk = timezone('Europe/Moscow')
    return dt.astimezone(msk)

app.jinja_env.filters['msk'] = to_moscow_time

# ========== МОДЕЛИ БАЗЫ ДАННЫХ ==========
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    wallet_balance = db.Column(db.Float, default=50000.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.email}>'


class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    short_description = db.Column(db.String(300))
    category = db.Column(db.String(50), default='Технологии')
    image_url = db.Column(db.String(500), nullable=True)
    goal_amount = db.Column(db.Float, nullable=False)
    current_amount = db.Column(db.Float, default=0)
    risk_score = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='draft')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    end_date = db.Column(db.DateTime, default=lambda: datetime.utcnow() + timedelta(days=30))

    creator = db.relationship('User', backref=db.backref('projects', lazy=True))

    def __repr__(self):
        return f'<Project {self.title}>'

    @property
    def funding_percentage(self):
        if self.goal_amount > 0:
            return min(int((self.current_amount / self.goal_amount) * 100), 100)
        return 0

    @property
    def risk_level(self):
        if self.risk_score >= 70:
            return {'level': 'Высокий', 'color': 'danger'}
        elif self.risk_score >= 40:
            return {'level': 'Средний', 'color': 'warning'}
        else:
            return {'level': 'Низкий', 'color': 'success'}

    @property
    def backers_count(self):
        return db.session.query(func.count(func.distinct(WalletTransaction.user_id)))\
            .filter(WalletTransaction.project_id == self.id,
                    WalletTransaction.transaction_type == 'support').scalar() or 0

    @property
    def days_left(self):
        if self.end_date:
            delta = self.end_date - datetime.utcnow()
            return max(0, delta.days)
        return 0


class Investment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), default='pending')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    investor = db.relationship('User', backref=db.backref('investments', lazy=True))
    project = db.relationship('Project', backref=db.backref('investments', lazy=True))


class WalletTransaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=True)
    amount = db.Column(db.Float, nullable=False)
    transaction_type = db.Column(db.String(20), nullable=False)
    description = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('wallet_transactions', lazy=True))
    project = db.relationship('Project', backref=db.backref('wallet_transactions', lazy=True))


# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ==========
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file):
    """Сохраняет файл и возвращает относительный URL"""
    if file and allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        return f'/static/uploads/{unique_filename}'
    return None


# ========== ML ОЦЕНКА РИСКОВ ==========
def calculate_risk_score(project_data):
    try:
        return risk_model.predict_risk_score(project_data)
    except Exception as e:
        print(f"⚠️ Ошибка ML модели: {e}")
        return 50


# ========== ДЕКОРАТОР АВТОРИЗАЦИИ ==========
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Необходимо войти в систему', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ========== ИНИЦИАЛИЗАЦИЯ БД И ТЕСТОВЫХ ДАННЫХ ==========
def create_tables():
    with app.app_context():
        db.create_all()

        if User.query.count() == 0:
            test_user = User(
                email='test@example.com',
                password_hash=generate_password_hash('password'),
                first_name='Тестовый',
                last_name='Пользователь',
                wallet_balance=50000.0
            )
            db.session.add(test_user)

            investor_user = User(
                email='investor@example.com',
                password_hash=generate_password_hash('password'),
                first_name='Иван',
                last_name='Инвестор',
                wallet_balance=75000.0
            )
            db.session.add(investor_user)
            db.session.commit()

            test_project = Project(
                user_id=test_user.id,
                title='Инновационное мобильное приложение',
                description='Разработка революционного мобильного приложения для управления финансами. '
                            'В команде есть опыт создания подобных продуктов, уже есть рабочий прототип '
                            'и проведено тестирование с потенциальными пользователями.',
                short_description='Мобильное приложение для управления финансами с готовым прототипом.',
                category='Технологии',
                image_url=None,
                goal_amount=25000,
                current_amount=7500,
                status='active',
                end_date=datetime.utcnow() + timedelta(days=30)
            )
            test_project.risk_score = calculate_risk_score({
                'title': test_project.title,
                'description': test_project.description,
                'goal_amount': test_project.goal_amount
            })
            db.session.add(test_project)
            db.session.commit()

            t1 = WalletTransaction(
                user_id=test_user.id,
                amount=50000.0,
                transaction_type='deposit',
                description='Начальный баланс кошелька'
            )
            t2 = WalletTransaction(
                user_id=investor_user.id,
                amount=75000.0,
                transaction_type='deposit',
                description='Начальный баланс кошелька'
            )
            db.session.add(t1)
            db.session.add(t2)
            db.session.commit()

            print("✅ Тестовые данные успешно созданы!")


# ========== ОСНОВНЫЕ МАРШРУТЫ ==========


@app.route('/')
def index():
    # Параметры фильтрации
    search_query = request.args.get('q', '').strip()
    category_filter = request.args.get('category', '').strip()

    # Базовый запрос: только активные проекты
    query = Project.query.filter_by(status='active')

    if search_query:
        query = query.filter(
            (Project.title.ilike(f'%{search_query}%')) |
            (Project.description.ilike(f'%{search_query}%')) |
            (Project.short_description.ilike(f'%{search_query}%'))
        )

    if category_filter:
        query = query.filter(Project.category == category_filter)

    projects = query.order_by(Project.created_at.desc()).all()

    # Список категорий для фильтра
    categories = db.session.query(Project.category)\
        .filter(Project.status == 'active')\
        .distinct().order_by(Project.category).all()
    categories = [c[0] for c in categories]

    # Статистика
    total_projects = Project.query.count()
    total_funding = db.session.query(db.func.sum(Project.current_amount)).scalar() or 0
    total_backers = db.session.query(func.count(func.distinct(WalletTransaction.user_id)))\
        .filter(WalletTransaction.transaction_type == 'support').scalar() or 0

    return render_template('index.html',
                           projects=projects,
                           categories=categories,
                           selected_category=category_filter,
                           search_query=search_query,
                           total_projects=total_projects,
                           total_funding=total_funding,
                           total_backers=total_backers)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        first_name = request.form['first_name']
        last_name = request.form['last_name']

        if User.query.filter_by(email=email).first():
            flash('Пользователь с таким email уже существует', 'danger')
            return render_template('register.html')

        user = User(
            email=email,
            password_hash=generate_password_hash(password),
            first_name=first_name,
            last_name=last_name,
            wallet_balance=50000.0
        )
        db.session.add(user)
        db.session.commit()

        bonus = WalletTransaction(
            user_id=user.id,
            amount=50000.0,
            transaction_type='deposit',
            description='Приветственный бонус'
        )
        db.session.add(bonus)
        db.session.commit()

        flash('Регистрация успешна! На ваш кошелёк зачислено $50,000', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['user_name'] = f"{user.first_name} {user.last_name}"
            flash(f'Добро пожаловать, {user.first_name}! Баланс: ${user.wallet_balance:.2f}', 'success')
            return redirect(url_for('index'))
        else:
            flash('Неверный email или пароль', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_name', None)
    flash('Вы вышли из системы', 'info')
    return redirect(url_for('index'))


@app.route('/create_project', methods=['GET', 'POST'])
@login_required
def create_project():
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        goal_amount = float(request.form['goal_amount'])
        category = request.form.get('category', 'Технологии')

        # Загрузка изображения
        image_url = None
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                image_url = save_uploaded_file(file)

        short_description = description[:250] + '...' if len(description) > 250 else description

        project_data = {'title': title, 'description': description, 'goal_amount': goal_amount}
        risk_score = calculate_risk_score(project_data)

        # Получаем длительность из формы (по умолчанию 30)
        try:
            duration = int(request.form.get('duration', 30))
        except ValueError:
            duration = 30

        # Ограничиваем разумными пределами
        if duration < 1:
            duration = 1
        if duration > 90:
            duration = 90

        project = Project(
            user_id=session['user_id'],
            title=title,
            description=description,
            short_description=short_description,
            category=category,
            image_url=image_url,
            goal_amount=goal_amount,
            risk_score=risk_score,
            status='active',
            end_date=datetime.utcnow() + timedelta(days=duration)   # теперь пользовательская длительность
            
        )

        db.session.add(project)
        db.session.commit()

        flash(f'Проект "{title}" создан! Риск-скор: {risk_score}/100', 'success')
        return redirect(url_for('project_detail', project_id=project.id))

    return render_template('create_project.html')


@app.route('/project/<int:project_id>')
def project_detail(project_id):
    project = Project.query.get_or_404(project_id)
    supports = WalletTransaction.query.filter_by(
        project_id=project_id,
        transaction_type='support'
    ).order_by(WalletTransaction.created_at.desc()).all()

    return render_template('project_detail.html',
                           project=project,
                           supports=supports)


# ========== МАРШРУТЫ КОШЕЛЬКА ==========
@app.route('/wallet')
@login_required
def wallet():
    user = User.query.get(session['user_id'])
    transactions = WalletTransaction.query.filter_by(user_id=session['user_id'])\
        .order_by(WalletTransaction.created_at.desc()).all()
    return render_template('wallet.html', user=user, transactions=transactions)


@app.route('/support/<int:project_id>', methods=['POST'])
@login_required
def support_project(project_id):
    try:
        amount = float(request.form['amount'])
        project = Project.query.get_or_404(project_id)
        user = User.query.get(session['user_id'])

        # Проверка: не закончился ли срок сбора
        if project.end_date and datetime.utcnow() > project.end_date:
            return jsonify({
                'success': False,
                'message': 'Кампания завершена. Поддержка больше не принимается.'})

        if user.wallet_balance < amount:
            return jsonify({'success': False,
                            'message': f'Недостаточно средств. Баланс: ${user.wallet_balance:.2f}'})

        if amount <= 0:
            return jsonify({'success': False, 'message': 'Сумма должна быть больше 0'})

        if project.user_id == session['user_id']:
            return jsonify({'success': False, 'message': 'Нельзя поддерживать собственные проекты'})

        transaction = WalletTransaction(
            user_id=session['user_id'],
            project_id=project_id,
            amount=amount,
            transaction_type='support',
            description=f'Поддержка проекта: {project.title}'
        )

        user.wallet_balance -= amount
        project.current_amount += amount

        db.session.add(transaction)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'Спасибо! Вы поддержали проект на ${amount:.2f}',
            'new_balance': user.wallet_balance,
            'project_amount': project.current_amount,
            'percentage': project.funding_percentage
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка: {str(e)}'})


@app.route('/add_funds', methods=['POST'])
@login_required
def add_funds():
    try:
        amount = float(request.form['amount'])
        user = User.query.get(session['user_id'])

        if amount <= 0:
            return jsonify({'success': False, 'message': 'Сумма должна быть больше 0'})

        transaction = WalletTransaction(
            user_id=session['user_id'],
            amount=amount,
            transaction_type='deposit',
            description='Пополнение кошелька'
        )
        user.wallet_balance += amount
        db.session.add(transaction)
        db.session.commit()

        return jsonify({'success': True,
                        'message': f'Кошелёк пополнен на ${amount:.2f}',
                        'new_balance': user.wallet_balance})

    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка: {str(e)}'})


@app.route('/my_projects')
@login_required
def my_projects():
    projects = Project.query.filter_by(user_id=session['user_id'])\
        .order_by(Project.created_at.desc()).all()
    return render_template('my_projects.html', projects=projects)


@app.route('/my_investments')
@login_required
def my_investments():
    investments = Investment.query.filter_by(user_id=session['user_id'])\
        .order_by(Investment.created_at.desc()).all()
    supports = WalletTransaction.query.filter_by(
        user_id=session['user_id'],
        transaction_type='support'
    ).order_by(WalletTransaction.created_at.desc()).all()
    return render_template('my_investments.html',
                           investments=investments,
                           supports=supports)


@app.route('/delete_project/<int:project_id>', methods=['POST'])
@login_required
def delete_project(project_id):
    project = Project.query.get_or_404(project_id)

    if project.user_id != session['user_id']:
        flash('У вас нет прав для удаления этого проекта', 'danger')
        return redirect(url_for('my_projects'))

    # Удаляем файл изображения
    if project.image_url and project.image_url.startswith('/static/uploads/'):
        file_path = os.path.join(app.root_path, project.image_url.lstrip('/'))
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Ошибка при удалении файла {file_path}: {e}")

    # Возврат средств
    supports = WalletTransaction.query.filter_by(
        project_id=project_id,
        transaction_type='support'
    ).all()

    for sup in supports:
        user = User.query.get(sup.user_id)
        user.wallet_balance += sup.amount
        refund = WalletTransaction(
            user_id=sup.user_id,
            project_id=project_id,
            amount=sup.amount,
            transaction_type='refund',
            description=f'Возврат средств за проект "{project.title}" (удалён)'
        )
        db.session.add(refund)
        db.session.delete(sup)

    WalletTransaction.query.filter_by(project_id=project_id).delete()
    Investment.query.filter_by(project_id=project_id).delete()

    db.session.delete(project)
    db.session.commit()

    flash(f'Проект "{project.title}" удалён, средства возвращены поддержавшим', 'success')
    return redirect(url_for('my_projects'))


# ========== АДМИНИСТРИРОВАНИЕ ==========
@app.route('/admin')
@login_required
def admin():
    user = User.query.get(session['user_id'])
    if user.email != 'test@example.com':
        flash('Доступ запрещён', 'danger')
        return redirect(url_for('index'))

    users = User.query.all()
    projects = Project.query.all()
    transactions = WalletTransaction.query.all()

    return render_template('admin.html',
                           users=users,
                           projects=projects,
                           transactions=transactions)


@app.route('/admin/deduct/<int:user_id>', methods=['POST'])
@login_required
def admin_deduct(user_id):
    admin_user = User.query.get(session['user_id'])
    if admin_user.email != 'test@example.com':
        flash('Нет прав для списания средств', 'danger')
        return redirect(url_for('admin'))

    user = User.query.get_or_404(user_id)
    amount = abs(float(request.form['amount']))

    if user.wallet_balance >= amount:
        user.wallet_balance -= amount
        txn = WalletTransaction(
            user_id=user.id,
            amount=amount,
            transaction_type='admin_deduct',
            description='Списание админом'
        )
        db.session.add(txn)
        db.session.commit()
        flash(f'Списано ${amount:.2f} с аккаунта {user.email}', 'success')
    else:
        flash(f'Недостаточно средств на балансе {user.email}. Баланс: ${user.wallet_balance:.2f}', 'danger')

    return redirect(url_for('admin'))


@app.route('/admin/recalculate_all_risks', methods=['POST'])
@login_required
def recalculate_all_risks():
    user = User.query.get(session['user_id'])
    if user.email != 'test@example.com':
        flash('Недостаточно прав', 'danger')
        return redirect(url_for('admin'))

    projects = Project.query.all()
    for project in projects:
        new_score = calculate_risk_score({
            'title': project.title,
            'description': project.description,
            'goal_amount': project.goal_amount,
            'current_amount': project.current_amount
        })
        project.risk_score = new_score

    db.session.commit()
    flash('Риски всех проектов пересчитаны!', 'success')
    return redirect(url_for('admin'))


@app.route('/api/user_balance')
@login_required
def get_user_balance():
    user = User.query.get(session['user_id'])
    return jsonify({'balance': user.wallet_balance})


# ========== ЗАПУСК ==========
if __name__ == '__main__':
    create_tables()
    print("💰 CrowdFunding Platform запущена")
    print("🌐 http://localhost:5000")
    app.run(debug=True, port=5000)