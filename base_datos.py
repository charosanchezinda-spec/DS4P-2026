from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
URL_BASE_DATOS = os.getenv("DATABASE_URL")

engine = create_engine(URL_BASE_DATOS)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class CorrridaDB(Base):
    __tablename__ = "corridas"

    id = Column(Integer, primary_key=True, index=True)
    fecha_hora = Column(String)
    poblacion = Column(String)
    n_registros = Column(Integer)
    variables_calib = Column(Text)

class MetricaDB(Base):
    __tablename__ = "metricas"

    id = Column(Integer, primary_key=True, index=True)
    corrida_id = Column(Integer)
    deff = Column(Float)
    ess = Column(Float)
    essp = Column(Float)
    peso_max = Column(Float)
    peso_min = Column(Float)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def registrar_corrida(db, poblacion, n_registros, variables_calib):
    corrida = CorrridaDB(
        fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        poblacion = poblacion,
        n_registros = n_registros,
        variables_calib = ", ".join(variables_calib)
    )
    db.add(corrida)
    db.commit()
    db.refresh(corrida)
    return corrida.id

def registrar_metricas(db, corrida_id, deff, ess, essp, peso_max, peso_min):
    metrica = MetricaDB(
        corrida_id = corrida_id,
        deff = deff,
        ess = ess,
        essp = essp,
        peso_max = peso_max,
        peso_min = peso_min
    )
    db.add(metrica)
    db.commit()
