import subprocess  # For running shell commands, 🚀🐚🚀
                   # With Python's touch, so grand! 🌟🐍🌟

# A function so neat, a treat to repeat, 🍬🎶🍬
def check_git_config():  # Let's take a seat! 🪑🌟🪑
    try:
        # For email, we'll peek, with Python technique! 📧🔍📧
        email = subprocess.check_output(
            ["git", "config", "--global", "user.email"],
            text=True).strip()
        # For name, the same, in this Git game! 🎮🔍🎮
        name = subprocess.check_output(
            ["git", "config", "--global", "user.name"],
            text=True).strip()

        # If found around, let joy resound! 🎉✨🎉
        if email and name:
            print(f"Email found: {email}, 📧🌈📧\nName's around: {name}! 🌟👤🌟")
            return True
        else:
            print("Some configs are missing, 🚫🤔🚫\nLet's keep on fishing! 🎣🌊🎣")
            return False
    except subprocess.CalledProcessError:
        # If error's in sight, we'll set it right! 🚨🛠️🚨
        print("Git configs not found, 🚫🔍🚫\nIn silence they're bound. 🤫🌌🤫")
        return None


if __name__ == "__main__":
    # Now let's invoke, with a stroke of hope! 🌈🙏🌈
    check_git_config()