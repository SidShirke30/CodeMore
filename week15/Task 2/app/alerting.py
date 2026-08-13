from datetime import datetime


def trigger_alert(alerts):
    if not alerts:
        return None

    message = (
        f"[{datetime.now().isoformat(timespec='seconds')}] "
        "MODEL PERFORMANCE ALERT: " + "; ".join(alerts)
    )
    print(message)
    return message
