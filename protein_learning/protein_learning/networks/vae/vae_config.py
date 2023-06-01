class VAEConfig:

    def __init__(
            self,
            scalar_dim_in : int,
            pair_dim_in: int,
            dim_latent: int,
            use_diag_cov: bool = True,
    ):
        self.scalar_dim_in = scalar_dim_in
        self.pair_dim_in = pair_dim_in
        self.dim_latent = dim_latent
        self.use_diag_cov = use_diag_cov
