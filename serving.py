import torch
import uvicorn

from caraml.core import model_loader
from pydantic import BaseModel
from fastapi import FastAPI

from enum import Enum


class Gender(Enum):
    MALE = 0
    FEMALE = 1
    OTHER = 2


class InsuranceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = torch.nn.Linear(2, 10)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(10, 2)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


@model_loader
def create_model_v1() -> InsuranceModel:
    return InsuranceModel()


@model_loader
def create_model_v2() -> InsuranceModel:
    return InsuranceModel()


def should_insure_v1(age: int, gender: Gender) -> float:
    gender_idx = gender.value

    model = create_model_v1()

    predictions: torch.Tensor = model(torch.tensor([age * 1.0, gender_idx * 1.0]))

    return predictions[0].item()


def should_insure_v2(age: int, gender: Gender) -> bool:
    gender_idx = gender.value

    model = create_model_v2()

    predictions: torch.Tensor = model(torch.tensor([age * 1.0, gender_idx * 1.0]))

    yes_pred, no_pred = predictions.tolist()

    return yes_pred > no_pred


def main():
    print("Trying a few examples...")

    for age, gender in [
        (12, Gender.MALE),
        (16, Gender.FEMALE),
        (42, Gender.OTHER)
    ]:
        v1_likelihood = should_insure_v1(age, gender)
        v2_pred = should_insure_v2(age, gender)
        print(f"  {age}, {gender} => v1: {v1_likelihood}, v2: {v2_pred}")


def serving_main():

    class Request(BaseModel):
        age: int
        gender: Gender

    class Response(BaseModel):
        should_insure: bool

    app = FastAPI()

    @app.get("/v1/{age}/{gender}")
    def v1_endpoint(age: int, gender: Gender):
        print(type(age), type(gender))
        result = should_insure_v1(age, gender) > 0.5
        return Response(
            should_insure=result
        )

    @app.post("/v2")
    def v2_endpoint(body: Request):
        return Response(
            should_insure=should_insure_v2(body.age, body.gender)
        )

    uvicorn.run(app, port=5000)


if __name__ == '__main__':
    main()
    serving_main()
