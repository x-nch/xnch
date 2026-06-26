'''Policy validator.

Class PolicyValidator has a method `validate(policy: dict) -> bool` which checks that the policy dict has required keys: name, rules, priority. Returns True if valid, False otherwise.
'''

class PolicyValidator:
    def validate(self, policy: dict) -> bool:
        required_keys = ['name', 'rules', 'priority']
        return all(key in policy for key in required_keys)