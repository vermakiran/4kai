# app/config.py
import os
from dotenv import load_dotenv

# Load .env from your project root (where you run python run.py)
load_dotenv()

class Settings:
    # Will be an empty string if JWT_SECRET is not set
    JWT_SECRET: str = os.getenv("JWT_SECRET", "")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    
    # new HANA settings
    HANA_HOST: str = os.getenv("HANA_HOST", "")
    HANA_PORT: int = int(os.getenv("HANA_PORT", "0"))
    HANA_USER: str = os.getenv("HANA_USER", "")
    HANA_PASS: str = os.getenv("HANA_PASS", "")
    HANA_SCHEMA: str = os.getenv("HANA_SCHEMA", "")

settings = Settings()

# Organization calendar configuration
class OrganizationCalendarConfig:
    """
    Configuration for organization-specific calendar settings.
    Different organizations may have different business calendars.
    """
    
    # Week start day mapping: organization_id -> week_start_day
    # Week start days: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday
    ORGANIZATION_WEEK_START = {
        'default': 0,        # Monday (ISO standard)
        'org_a': 0,         # Monday to Sunday (example: Org A)
        'org_b': 5,         # Saturday to Friday (example: Org B)
        'org_c': 6,         # Sunday to Saturday (example: Org C)
    }
    
    # Pandas period mapping for week start days
    WEEK_START_PANDAS_MAP = {
        0: 'W-MON',    # Monday
        1: 'W-TUE',    # Tuesday  
        2: 'W-WED',    # Wednesday
        3: 'W-THU',    # Thursday
        4: 'W-FRI',    # Friday
        5: 'W-SAT',    # Saturday
        6: 'W-SUN',    # Sunday
    }
    
    @classmethod
    def get_week_start_day(cls, organization_id='default'):
        """
        Get the week start day for an organization.
        
        Args:
            organization_id (str): Organization identifier
            
        Returns:
            int: Week start day (0=Monday, 6=Sunday)
        """
        return cls.ORGANIZATION_WEEK_START.get(organization_id, 0)
    
    @classmethod
    def get_pandas_week_rule(cls, organization_id='default'):
        """
        Get the pandas week rule for an organization.
        
        Args:
            organization_id (str): Organization identifier
            
        Returns:
            str: Pandas week rule (e.g., 'W-MON', 'W-SAT')
        """
        week_start = cls.get_week_start_day(organization_id)
        return cls.WEEK_START_PANDAS_MAP.get(week_start, 'W-MON')
    
    @classmethod
    def add_organization_calendar(cls, organization_id, week_start_day):
        """
        Add or update organization calendar configuration.
        
        Args:
            organization_id (str): Organization identifier
            week_start_day (int): Week start day (0=Monday, 6=Sunday)
        """
        if not (0 <= week_start_day <= 6):
            raise ValueError("Week start day must be between 0 (Monday) and 6 (Sunday)")
        
        cls.ORGANIZATION_WEEK_START[organization_id] = week_start_day
    
    @classmethod
    def get_week_start_name(cls, organization_id='default'):
        """
        Get the human-readable name of the week start day.
        
        Args:
            organization_id (str): Organization identifier
            
        Returns:
            str: Week start day name (e.g., 'Monday', 'Saturday')
        """
        week_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        week_start = cls.get_week_start_day(organization_id)
        return week_names[week_start]
