"""
Episodic Memory V3 — SQLite Storage
Tracks structured history of sessions, events, and metrics.
"""
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import os
from config import settings

Base = declarative_base()

class Driver(Base):
    __tablename__ = 'drivers'
    id = Column(String, primary_key=True)
    baseline_ear = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    sessions = relationship("Session", back_populates="driver")

class Session(Base):
    __tablename__ = 'sessions'
    id = Column(String, primary_key=True)
    driver_id = Column(String, ForeignKey('drivers.id'))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    driver = relationship("Driver", back_populates="sessions")
    events = relationship("SafetyEvent", back_populates="session")

class SafetyEvent(Base):
    __tablename__ = 'safety_events'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('sessions.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    event_type = Column(String)  # drowsy, distracted, phone, head_nod, yawn
    severity = Column(Integer)   # 1-10
    agent_handled = Column(Boolean, default=False)
    session = relationship("Session", back_populates="events")

class EpisodicMemory:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = settings.memory_db_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_or_create_driver(self, driver_id: str) -> Driver:
        with self.SessionLocal() as db:
            driver = db.query(Driver).filter(Driver.id == driver_id).first()
            if not driver:
                driver = Driver(id=driver_id)
                db.add(driver)
                db.commit()
                db.refresh(driver)
            # Create a detached copy to return
            db.expunge(driver)
            return driver

    def create_session(self, session_id: str, driver_id: str):
        with self.SessionLocal() as db:
            session = Session(id=session_id, driver_id=driver_id)
            db.add(session)
            db.commit()

    def end_session(self, session_id: str):
        with self.SessionLocal() as db:
            session = db.query(Session).filter(Session.id == session_id).first()
            if session:
                session.end_time = datetime.utcnow()
                db.commit()

    def log_event(self, session_id: str, event_type: str, severity: int) -> int:
        with self.SessionLocal() as db:
            event = SafetyEvent(session_id=session_id, event_type=event_type, severity=severity)
            db.add(event)
            db.commit()
            db.refresh(event)
            return event.id

    def get_recent_events(self, session_id: str, limit: int = 10) -> list:
        with self.SessionLocal() as db:
            events = db.query(SafetyEvent)\
                       .filter(SafetyEvent.session_id == session_id)\
                       .order_by(SafetyEvent.timestamp.desc())\
                       .limit(limit)\
                       .all()
            
            # Convert to dicts before closing session
            return [
                {
                    "id": e.id,
                    "type": e.event_type,
                    "severity": e.severity,
                    "timestamp": e.timestamp.isoformat(),
                    "agent_handled": e.agent_handled
                } for e in events
            ]
