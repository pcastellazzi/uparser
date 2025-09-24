from typing import assert_never, cast

import uparser as p

TEXT = r"[\x20-\x21\x23-\x2B\x2D-\x7E]"


def strip_quotes(text: str) -> str:
    return text[1:-1].replace('""', '"') if text.startswith('"') else text


a_comma = p.atom(",")
a_crlf = p.atom("\r\n")
n_crlf = p.regex(r"(?:\r\n)*")

field = p.regex(rf'{TEXT}+|"(?:{TEXT}|""|,|\r\n)*"')
field = p.map_value(field, strip_quotes)
field = p.set_error(field, "expected a field")

a_field = p.map_value(field, lambda v: [v])
n_field = p.many0(p.skip1(a_comma, field))
record = p.sequence(a_field, n_field)
record = p.map_value(record, lambda v: v[0] + v[1])
record = p.set_error(record, "expected a record")

eof = p.sequence(n_crlf, p.eof())
eof = p.set_value(eof, cast("list[list[str]]", []))

csv = p.map_value(
    p.sequence(
        p.map_value(record, lambda v: [v]),
        p.many0(p.skip1(a_crlf, record)),
        eof,
    ),
    lambda v: v[0] + v[1],
)


def csv_parser(csv_string: str) -> list[list[str]]:
    match csv(0, csv_string):
        case p.Success(_, value):
            return value
        case p.Failure(index, error):
            message = f"Error: {error} at position {index}"
            raise ValueError(message)
        case _ as other:
            assert_never(other)


if __name__ == "__main__":
    csv_data = (
        "EmployeeID,FirstName,LastName,Email,Department\r\n"
        "101,John,Doe,john.doe@example.com,Engineering\r\n"
        "102,Jane,Smith,jane.smith@example.com,Marketing\r\n"
        "103,Peter,Jones,peter.jones@example.com,Sales\r\n"
        "104,Mary,O'Connell,mary.oconnell@example.com,HR\r\n"
        "105,David,Chen,david.chen@example.com,Engineering\r\n"
        '106,Susan,Williams,susan.williams@example.com,"Finance, Accounting"\r\n'
        "107,Michael,Brown,michael.brown@example.com,IT\r\n"
        "108,Linda,Davis,linda.davis@example.com,Sales\r\n"
        '109,James,Wilson,james.wilson@example.com,"Legal ""Corp"""\r\n'
    )
    records = csv_parser(csv_data)
    print(f"RESULT: {records!r}")
