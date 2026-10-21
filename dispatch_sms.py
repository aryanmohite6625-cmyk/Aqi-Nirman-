import os

# Set these via env variables or leave empty for mock testing
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

def evaluate_dispatch_rules(ward_row):
    """
    Evaluates unit allocation and urgency level based on AQI and hazard classification.
    """
    aqi = ward_row["Current_AQI"]
    spike_type = ward_row["Spike_Type"]
    radius = ward_row["Impact_Radius_KM"]

    if spike_type == "Chemical / Industrial Venting":
        units = ["1x MPCB Flying Squad (Hazardous Gas Unit)", "4x TTC Industrial Misting Cannons", "Ward Health Rapid Response"]
        priority = "CRITICAL (PRIORITY 1)"
    elif spike_type == "Particulate Dust Surge":
        cannons_needed = max(2, int(radius * 1.5))
        units = [f"{cannons_needed}x Anti-Smog Water Misting Trucks", "1x Construction Site Compliance Squad"]
        priority = "HIGH (PRIORITY 2)"
    elif spike_type == "Vehicular Exhaust Plume":
        units = ["Navi Mumbai Traffic Police Corridors Team", "1x Mobile Air Purifier Truck"]
        priority = "MEDIUM (PRIORITY 3)"
    else:
        units = ["Routine Ward Patrolling"]
        priority = "LOW"

    return {
        "priority": priority,
        "units": units,
        "ward": ward_row["Ward_Name"],
        "aqi": aqi,
        "spike_type": spike_type
    }

def send_dispatch_sms(dispatch_details, recipient_number="+919876543210"):
    """
    Dispatches SMS to field teams. Uses mock fall-back if Twilio credentials are missing.
    """
    body = (
        f"🚨 [NMMC DISPATCH - {dispatch_details['priority']}]\n"
        f"Ward: {dispatch_details['ward']}\n"
        f"Hazard: {dispatch_details['spike_type']} (AQI: {dispatch_details['aqi']})\n"
        f"Allocated Units:\n" + "\n".join([f"- {u}" for u in dispatch_details['units']]) + "\n"
        f"Immediate response required."
    )

    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER:
        try:
            from twilio.rest import Client
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            message = client.messages.create(
                body=body,
                from_=TWILIO_PHONE_NUMBER,
                to=recipient_number
            )
            return True, f"Live SMS sent via Twilio (SID: {message.sid})"
        except Exception as e:
            return False, f"Twilio SMS failed: {str(e)}"
    else:
        # Dev / Exhibition Demo Mock
        return True, f"Mock SMS Dispatched to {recipient_number}:\n\n{body}"
