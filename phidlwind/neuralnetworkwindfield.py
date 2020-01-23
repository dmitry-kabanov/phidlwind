import numpy as np
from geopandas import GeoDataFrame

import framework.windfield as wf
import framework.tools as tools

from phidlwind.neuralnetwork import DivFreeNeuralNetwork


class NeuralNetworkWindfield(wf.Windfield):
    """Wind predictor based on neural-network model.

    This class implements Adapter design pattern to conform to the expectations
    of the framework that provides geospatial wind data.

    Parameters
    ----------
    calibration_data : GeoDataFrame
        Measurements data.

    """

    def __init__(self, calibration_data: GeoDataFrame, params: dict = None):
        x1 = tools.get_x(calibration_data).values
        x2 = tools.get_y(calibration_data).values
        u1 = tools.get_u(calibration_data).values
        u2 = tools.get_v(calibration_data).values

        X_train = np.hstack((x1[:, None], x2[:, None]))
        y_train = np.hstack((u1[:, None], u2[:, None]))

        if params is None:
            params = {}
            params["gamma"] = 0.001
            params["epochs"] = 150

        self.nn = DivFreeNeuralNetwork(X_train, y_train, params["gamma"])
        self.nn.train(params["epochs"])

    def get_wind(self, x, y) -> wf.WindVector:
        """Predict wind at point :math:`(x, y)`."""
        # Find 2D velocity vector :math:`(u, v)`.
        X_new = np.array([[x, y]])
        prediction = self.nn.predict(X_new)

        u, v = prediction[0, 0], prediction[0, 1]

        # Return wind vector.
        return tools.create_wind_vector(x, y, u, v)
