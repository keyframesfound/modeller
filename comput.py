    import numpy as np
    from scipy import stats

    # Input data
    t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 , 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])  # Time (months)
    N = np.array([
        557,
        27644,
        98078,
        1254222,
        3680614,
        6789073,
        11474058,
        18817443,
        26944997,
        35551524,
        48820295,
        66706697,
        86765351,
        105766676,
        116546593,
        132302655,
        155692589,
        173489803,
        184670098,
        201444210,
        221347788,
        236397732,
        249839415,
        266146231,
        298385756,
        393779507,
        445550401,
        494394436,
        516684038,
        532530338,
        551700794,
        583199840,
        605881148,
        620005197,
        632744972,
        645774500,
        662966595,
        671721476,
        676024901,
        676570149
    ])  # Cumulative cases
    K_candidates = [700_000_000, 750_000_000, 800_000_000, 850_000_000, 900_000_000, 950_000_000, 1_000_000_000, 1_500_000_000, 2_000_000_000]  # Candidate K values

    best_R2 = -np.inf
    best_K = None
    best_lambda = None

    for K in K_candidates:
        # Transform data: y = ln(N / (K - N))
        y = np.log(N / (K - N))
        
        # Perform linear regression
        slope, intercept, r_value, _, _ = stats.linregress(t, y)
        R2 = r_value ** 2
        
        # Check if this K gives a better R2
        if R2 > best_R2:
            best_R2 = R2
            best_K = K
            best_lambda = slope

    # Output results
    print(f"Best K: {best_K}")
    print(f"Best λ (growth rate): {best_lambda}")
    print(f"Best R²: {best_R2}")