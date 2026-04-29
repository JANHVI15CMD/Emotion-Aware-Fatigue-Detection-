def get_fatigue_level(score):

    if score < 0.5:
        return "Fresh 😊"

    elif score < 0.7:
        return "Slightly Tired 😐"

    elif score < 0.8:
        return "Moderately Fatigued 😓"

    else:
        return "Exhausted 😴"


# 🔹 Test
if __name__ == "__main__":
    test_scores = [0.2, 0.4, 0.6, 0.9]

    for s in test_scores:
        print(f"Score: {s} → {get_fatigue_level(s)}")
