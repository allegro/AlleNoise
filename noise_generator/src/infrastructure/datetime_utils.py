from datetime import datetime

from dateutil.relativedelta import relativedelta


DATE_FORMAT = "%Y-%m-%d"


def date_to_str(date: datetime) -> str:
    return datetime.strftime(date, DATE_FORMAT)


def yesterday() -> str:
    yesterday = datetime.now() + relativedelta(days=-1)
    return date_to_str(yesterday)
