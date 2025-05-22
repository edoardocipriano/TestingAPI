from pydantic import BaseModel, Field

class InputData(BaseModel):
    age: float = Field(..., description="Age of the patient")
    gender: str = Field(..., description="Gender of the patient (Female, Male, or Other)")
    hypertension: int = Field(..., description="Hypertension status (0 or 1)")
    heart_disease: int = Field(..., description="Heart disease status (0 or 1)")
    smoking_history: str = Field(..., description="Smoking history (No Info, never, former, not current, current, ever)")
    bmi: float = Field(..., description="Body mass index")
    hba1c_level: float = Field(..., description="HbA1c level")
    blood_glucose_level: float = Field(..., description="Blood glucose level")

class OutputData(BaseModel):
    diabetes: str = Field(..., description="Diabetes prediction (Yes or No)")
    probability: float = Field(..., description="Probability of diabetes")