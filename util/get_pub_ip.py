import requests

def get_public_ip_with_fallback():
    services = [
        "https://api.ipify.org",
        "https://ifconfig.me",
        "https://ident.me",
        "https://checkip.amazonaws.com"
    ]
    
    for service in services:
        try:
            response = requests.get(service, timeout=3)
            ip = response.text.strip()
            if response.status_code == 200 and valid_ip(ip):
                print(f"✅ Successfully got public IP: {ip} from {service}.")
                return ip
        except Exception:
            continue
    return "Failed to fetch public IP QAQ. "

def valid_ip(ip: str) -> bool:
    parts = ip.split(".")
    return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)

print("public IP:", get_public_ip_with_fallback())