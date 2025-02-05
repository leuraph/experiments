def main() -> None:
    max_n_dofs: int = int(1e7)
    n_initial_refinement_steps: int = 5

    # read the initial data
    # ---------------------
    

    # perform initial refinement to get a decent mesh
    # -----------------------------------------------
    for _ in range(n_initial_refinement_steps):
        pass

    # solve problem on initial mesh
    # -----------------------------

    # initialize the iteration with Galerkin solution on initial mesh
    # ---------------------------------------------------------------

    # perform initial EVA by hand
    # ---------------------------

    # loop until maximum number of degrees of freedom is reached
    # ----------------------------------------------------------
    while True:
        # recalculate mesh specific objects / parameters
        # ----------------------------------------------

        # calculate the Galerkin solution on the current mesh
        # ---------------------------------------------------

        # perform iterations until stopping criterion is met
        # --------------------------------------------------

        # drop all the data accumulated in the corresponding results directory
        # --------------------------------------------------------------------

        # perform EVA with the last iterate
        # ---------------------------------

        # calculate the number of degrees of freedom on the new mesh
        # ----------------------------------------------------------
        n_dofs: int = int(1e8)
        if n_dofs > max_n_dofs:
            break


if __name__ == '__main__':
    main()
