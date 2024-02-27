from smt.surrogate_models.surrogate_model import SurrogateModel
import egobox as egx


class GPX(SurrogateModel):
    name = "GPX"

    def _initialize(self):
        super(GPX, self)._initialize()

        self.supports["variances"] = True

    def _train(self):
        xt, yt = self.training_points[None][0]
        self.gpx = egx.GpMix().fit(xt, yt)

    def _predict_values(self, xt):
        y = self.gpx.predict_values(xt)
        return y

    def predict_variances(self, xt):
        s2 = self.gpx.predict_variances(xt)
        return s2
