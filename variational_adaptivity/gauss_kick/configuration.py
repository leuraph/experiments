from pydantic import BaseModel


class GaussKickExperimentConfiguration(BaseModel):
    """configuration for the gauss kick function"""
    sigma_x: float
    sigma_y: float
    mu_x_numerator: float
    mu_x_denominator: float
    mu_y_numerator: float
    mu_y_denomiator: float
    n_initial_refinements: int
    n_refinements: int

    @property
    def mu_x(self) -> float:
        return self.mu_x_numerator/self.mu_x_denominator

    @property
    def mu_y(self) -> float:
        return self.mu_y_numerator/self.mu_y_denomiator
