import click


@click.command()
def predict():
    print("predict")


@click.command()
def relax():
    print("relax")


if __name__ == "__main__":
    predict()
