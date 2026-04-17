# Source - https://stackoverflow.com/q/45400284
# Posted by EsotericVoid, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-17, License - CC BY-SA 3.0

class Philosopher:
    def __init_subclass__(cls, default_name, **kwargs):
        super().__init_subclass__(**kwargs)
        print(f"Called __init_subclass({cls}, {default_name})")
        cls.default_name = default_name

class AustralianPhilosopher(Philosopher, default_name="Bruce"):
    pass

class GermanPhilosopher(Philosopher, default_name="Nietzsche"):
    default_name = "Hegel"
    print("Set name to Hegel")

Bruce = AustralianPhilosopher()
Mistery = GermanPhilosopher()
print(Bruce.default_name)
print(Mistery.default_name)
