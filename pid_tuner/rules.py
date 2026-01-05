def get_dynamic_recommendation(diagnosis, axis_name, severity=0.0):
    """
    diagnosis: str (e.g., "P_High")
    axis_name: str (e.g., "Roll")
    severity: float (Calculated from features, 0.0 to ~1.0+)
    """
    diagnosis = diagnosis.strip()
    
    if diagnosis == 'Good':
        return f"{axis_name} is well-tuned. Good job!"

    # --- P TERM RULES ---
    if diagnosis == 'P_High':
        # Severity = Spectral Purity (0.2 = messy, 0.8 = pure sine wave)
        if severity > 0.5:
            drop = "15-20%"
            tone = "Major oscillations detected."
        elif severity > 0.2:
            drop = "10%"
            tone = "Moderate oscillations."
        else:
            drop = "5%"
            tone = "Minor flutter."
        return f"Reduce P on {axis_name} by {drop}. {tone}"

    if diagnosis == 'P_Low':
        # Severity = Overshoot Ratio (0.1 = small, 0.5 = huge overshoot)
        if severity > 0.4:
            boost = "20-30%"
            tone = "Tracking is very loose."
        elif severity > 0.2:
            boost = "10-15%"
            tone = "Response is sluggish."
        else:
            boost = "5%"
            tone = "Slightly sharper feel needed."
        return f"Increase P on {axis_name} by {boost}. {tone}"

    # --- D TERM RULES ---
    if diagnosis == 'D_High':
        # Severity = Noise Ratio (1.0 = D is louder than Gyro)
        if severity > 1.5:
            action = "Drop D by 20% immediately"
            tone = "D-term noise is critical (Hot motors risk)."
        elif severity > 0.8:
            action = "Reduce D by 10%"
            tone = "Significant noise visible."
        else:
            action = "Reduce D slightly"
            tone = "Minor noise floor issue."
        return f"{action} on {axis_name}. {tone}"

    if diagnosis == 'D_Low':
        return f"Increase D on {axis_name} to dampen propwash. Check motor heat after."

    # --- I TERM RULES ---
    if diagnosis == 'I_Low':
        if severity > 0.3:
            return f"Increase I on {axis_name} significantly. Drone is not holding angle."
        return f"Increase I on {axis_name} slightly to improve lock."

    if diagnosis == 'I_High':
        return f"Reduce I on {axis_name}. Low frequency wobble detected."

    return f"Check {axis_name} manually. (Diagnosis: {diagnosis})"