# TODO TESTER ET LINKER

def implied_volatility_call(C, S, K, T, r, q, tol=0.0001,
                            max_iterations=100):
    """"
    :param C: Observed call price
    :param S: Asset price
    :param K: Strike Price
    :param T: Time to Maturity
    :param r: riskfree rate
    :param q: dividend yield
    :param tol: error tolerance in result
    :param max_iterations: max iterations to update vol
    :return: implied volatility in percent
    """
    # vol initial value
    # TODO : better star version is sigma = sqrt(2 pi /T) * C/S
    sigma = np.sqrt(2 * np.pi / T) * C / S
    # sigma = 0.3

    for i in range(max_iterations):

        ### calculate difference between blackscholes price and market price with
        ### iteratively updated volality estimate
        diff = self.value(S, K, r, T, q, sigma, "call") - C

        ### break if difference is less than specified tolerance level
        if abs(diff) < tol:
            print(f'found on {i}th iteration')
            print(f'difference is equal to {diff}')
            break

        ### use newton rapshon to update the estimate
        sigma = sigma - diff / vega(S, K, T, r, q, sigma)

    return sigma