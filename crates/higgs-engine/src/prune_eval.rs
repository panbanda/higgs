//! Diverse, multi-step reasoning problem set + answer grader for the KV-prune
//! accuracy sweep.
//!
//! The problems are deliberately *not shallow*: each needs two or more dependent
//! steps, so pruning mid-trace context can plausibly break the chain. They span
//! nine categories (arithmetic, relational, algebra, counting, date/time, unit
//! conversion, sequences, logic, multi-fact comprehension) so the sweep measures
//! a general reasoning-degradation curve, not one narrow failure mode.
//!
//! Grading extracts the model's final answer (`Answer:` / `\boxed{}` / `####` /
//! last-number fallback) and compares it to the gold answer — numerically when
//! the gold is a number, else by normalized substring.

/// A single graded reasoning problem.
#[derive(Debug, Clone)]
pub struct Problem {
    /// Short category tag (for per-category breakdowns).
    pub category: &'static str,
    /// The question text.
    pub question: &'static str,
    /// Canonical gold answer (number like "72", or a word like "Dan").
    pub answer: &'static str,
}

/// The user-message content for a problem: question + answer-format instruction.
#[must_use]
pub fn prompt_for(problem: &Problem) -> String {
    format!(
        "{} Reason step by step, then end with a line in the exact form 'Answer: <answer>'.",
        problem.question
    )
}

/// 50 diverse, multi-step problems with verified short answers.
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn problem_set() -> Vec<Problem> {
    const P: fn(&'static str, &'static str, &'static str) -> Problem = make;
    vec![
        // --- A. Multi-step arithmetic word problems ---
        P(
            "arith",
            "Natalia sold clips to 48 of her friends in April, then sold half as many clips in May. How many clips did she sell altogether in April and May?",
            "72",
        ),
        P(
            "arith",
            "A bakery makes 120 loaves. They sell three quarters of them in the morning and 15 more in the afternoon. How many loaves are left?",
            "15",
        ),
        P(
            "arith",
            "Tom has $200. He buys 4 books at $18 each and a bag for $35. How much money does he have left, in dollars?",
            "93",
        ),
        P(
            "arith",
            "A train travels at 60 mph for 2.5 hours, then at 40 mph for 1.5 hours. How many miles does it travel in total?",
            "210",
        ),
        P(
            "arith",
            "A class has 30 students and 40% of them are boys. If 6 of the girls are absent today, how many girls are present?",
            "12",
        ),
        P(
            "arith",
            "Sarah earns $15 per hour and works 8 hours a day for 5 days. She saves 20% of her total earnings. How many dollars does she save?",
            "120",
        ),
        P(
            "arith",
            "A tank holds 500 liters. It is currently three fifths full, then 60 liters are added. How many liters are in the tank now?",
            "360",
        ),
        P(
            "arith",
            "A store had 80 apples. They received 3 boxes of 25 apples each, then sold 90 apples. How many apples remain?",
            "65",
        ),
        // --- B. Multi-hop relational reasoning ---
        P(
            "relational",
            "Anna is taller than Beth. Beth is taller than Carol. Carol is taller than Dan. Who is the shortest of the four?",
            "Dan",
        ),
        P(
            "relational",
            "Tom is Mary's son. Mary is John's sister. What relation is John to Tom?",
            "uncle",
        ),
        P(
            "relational",
            "Four boxes weigh 5, 8, 3, and 10 kg. The heaviest box and the lightest box are removed. What is the combined weight, in kg, of the two boxes that remain?",
            "13",
        ),
        P(
            "relational",
            "In a race, Maria finished ahead of Nina, Nina ahead of Olga, and Olga ahead of Pia. How many runners finished between Maria and Pia?",
            "2",
        ),
        P(
            "relational",
            "Five runners finished a race: Joe before Kim, Kim before Lee, Lee before Max, and Max before Ned. Who came in third place?",
            "Lee",
        ),
        P(
            "relational",
            "If the day before yesterday was Sunday, what day of the week is tomorrow?",
            "Wednesday",
        ),
        // --- C. Algebra / solve-for-x ---
        P("algebra", "If 3x + 7 = 22, what is the value of x?", "5"),
        P(
            "algebra",
            "Two numbers add up to 30 and differ by 8. What is the larger number?",
            "19",
        ),
        P(
            "algebra",
            "A rectangle's length is twice its width and its perimeter is 36. What is the width?",
            "6",
        ),
        P(
            "algebra",
            "If 5 pencils cost $2.00, how much do 12 pencils cost, in dollars?",
            "4.80",
        ),
        P(
            "algebra",
            "x is 3 more than y. y is twice z. If z = 4, what is x?",
            "11",
        ),
        P(
            "algebra",
            "The average of four numbers is 15. Three of them are 12, 18, and 10. What is the fourth number?",
            "20",
        ),
        // --- D. Combinatorics / counting ---
        P(
            "counting",
            "How many different 3-digit numbers can be formed using the digits 1, 2, 3, 4, 5 without repeating any digit?",
            "60",
        ),
        P(
            "counting",
            "In how many different orders can 4 people sit in a row?",
            "24",
        ),
        P(
            "counting",
            "A pizza comes in 3 sizes and has 4 topping choices. If you pick one size and one topping, how many different pizzas are possible?",
            "12",
        ),
        P(
            "counting",
            "How many diagonals does a hexagon (6-sided polygon) have?",
            "9",
        ),
        P(
            "counting",
            "How many even numbers are there between 1 and 50, inclusive?",
            "25",
        ),
        // --- E. Date / time arithmetic ---
        P(
            "datetime",
            "A movie starts at 7:45 PM and lasts 2 hours and 40 minutes. How many minutes after 10:00 PM does it end?",
            "25",
        ),
        P(
            "datetime",
            "If today is Wednesday, what day of the week will it be in 10 days?",
            "Saturday",
        ),
        P(
            "datetime",
            "A 30-day project starts on March 1, counting March 1 as day 1. On what day of the month does it end?",
            "30",
        ),
        P(
            "datetime",
            "How many minutes are there from 9:50 AM to 1:15 PM on the same day?",
            "205",
        ),
        P(
            "datetime",
            "On a 12-hour clock at 3:00, what is the smaller angle between the hour and minute hands, in degrees?",
            "90",
        ),
        // --- F. Unit conversion chains ---
        P(
            "units",
            "A car uses 8 liters of fuel per 100 km. How many liters does it use on a 250 km trip?",
            "20",
        ),
        P("units", "How many seconds are there in 2.5 hours?", "9000"),
        P(
            "units",
            "A recipe needs 250 grams of flour per cake. How many kilograms of flour are needed for 12 cakes?",
            "3",
        ),
        P(
            "units",
            "If 1 inch equals 2.54 cm, how many centimeters are in 10 inches?",
            "25.4",
        ),
        P(
            "units",
            "A tank fills at 5 liters per minute. How many liters does it hold after 1.5 hours?",
            "450",
        ),
        // --- G. Sequences / patterns ---
        P(
            "sequence",
            "What is the next number in the sequence 2, 6, 12, 20, 30, ...?",
            "42",
        ),
        P(
            "sequence",
            "What is the 6th term of the sequence that starts 3, 6, 12, 24, ... (each term doubles)?",
            "96",
        ),
        P(
            "sequence",
            "In the Fibonacci sequence 1, 1, 2, 3, 5, 8, ..., what is the 9th term?",
            "34",
        ),
        P(
            "sequence",
            "What is the next number in the sequence 1, 4, 9, 16, 25, ...?",
            "36",
        ),
        P(
            "sequence",
            "What is the sum of the first 10 positive integers?",
            "55",
        ),
        // --- H. Logic puzzles ---
        P(
            "logic",
            "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How many cents does the ball cost?",
            "5",
        ),
        P(
            "logic",
            "All Bloops are Razzies and all Razzies are Lazzies. Are all Bloops definitely Lazzies? Answer yes or no.",
            "yes",
        ),
        P(
            "logic",
            "A farmer has only chickens and cows. Together they have 10 heads and 28 legs. How many cows are there?",
            "4",
        ),
        P(
            "logic",
            "A snail climbs 3 feet up a well each day and slips back 2 feet each night. The well is 10 feet deep. On which day does it first reach the top?",
            "8",
        ),
        P(
            "logic",
            "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes does it take 100 machines to make 100 widgets?",
            "5",
        ),
        // --- I. Multi-fact comprehension (attend back over a passage) ---
        P(
            "comprehension",
            "A library had 1,200 books. In January it added 150 books, in February it removed 80 damaged books, and in March it donated 200 books to schools. How many books does the library have now?",
            "1070",
        ),
        P(
            "comprehension",
            "Maria's garden has 3 rows of tomatoes with 8 plants each, and 2 rows of peppers with 6 plants each. Each tomato plant yields 5 tomatoes. How many tomatoes does she harvest in total?",
            "120",
        ),
        P(
            "comprehension",
            "A company has three departments. Sales has 12 people, Engineering has twice as many people as Sales, and HR has 5 fewer people than Engineering. How many people work at the company in total?",
            "55",
        ),
        P(
            "comprehension",
            "On a road trip a family drove 320 km on the first day, 280 km on the second day, and on the third day they drove 50 km less than the second day. What was the total distance, in km?",
            "830",
        ),
        P(
            "comprehension",
            "A theater has 25 rows with 18 seats each. For tonight's show, 30 seats are reserved and 12 seats are broken. How many seats are available to sell?",
            "408",
        ),
    ]
}

const fn make(category: &'static str, question: &'static str, answer: &'static str) -> Problem {
    Problem {
        category,
        question,
        answer,
    }
}

/// One aggregated row of a prune-rate sweep.
#[derive(Debug, Clone)]
pub struct SweepRow {
    /// Target prune percentage for this row.
    pub prune_pct: u32,
    /// Fraction of problems graded correct.
    pub accuracy: f32,
    /// Mean peak resident KV (tokens) over the problems.
    pub mean_peak_kv: f32,
    /// Mean decode tokens/sec over the problems.
    pub mean_tok_per_s: f32,
    /// Number of problems in this row.
    pub n: u32,
}

/// Render sweep rows as a fixed-width table plus a one-line knee summary: the
/// highest prune rate whose accuracy stays within `tol` of the 0% row.
#[must_use]
pub fn render_table(rows: &[SweepRow], tol: f32) -> String {
    use std::fmt::Write as _;
    let mut out =
        String::from("prune% |  acc | peakKV | tok/s |  n\n-------+------+--------+-------+----\n");
    for r in rows {
        let _ = writeln!(
            out,
            "{:>5}% | {:>3.0}% | {:>6.0} | {:>5.1} | {:>2}",
            r.prune_pct,
            r.accuracy * 100.0,
            r.mean_peak_kv,
            r.mean_tok_per_s,
            r.n
        );
    }
    let baseline = rows
        .iter()
        .find(|r| r.prune_pct == 0)
        .map_or(0.0, |r| r.accuracy);
    let knee = rows
        .iter()
        .filter(|r| r.accuracy + tol >= baseline)
        .map(|r| r.prune_pct)
        .max()
        .unwrap_or(0);
    let _ = writeln!(
        out,
        "knee: accuracy holds (within {:.0}%) up to ~{knee}% prune",
        tol * 100.0
    );
    out
}

// --- Grading ------------------------------------------------------------------

/// Grade a model's free-form output against the gold answer.
#[allow(clippy::option_if_let_else)] // the match reads clearer than map_or_else here
#[must_use]
pub fn grade(output: &str, gold: &str) -> bool {
    let (region, had_marker) = answer_region(output);
    match parse_number(gold) {
        Some(gold_num) => {
            // With an explicit marker the answer is the last number on that line;
            // without one, fall back to the final number anywhere in the trace.
            let candidate = had_marker
                .then(|| last_number(&region))
                .flatten()
                .or_else(|| last_number(output));
            candidate.is_some_and(|n| (n - gold_num).abs() < 1e-6)
        }
        None => normalize_word(&region).contains(&normalize_word(gold)),
    }
}

/// Extract the model's final numeric answer (after the last marker, else the
/// last number in the trace). Used to carry a running value across self-summary
/// checkpoints.
#[must_use]
pub fn extract_number(text: &str) -> Option<f64> {
    let (region, had_marker) = answer_region(text);
    had_marker
        .then(|| last_number(&region))
        .flatten()
        .or_else(|| last_number(text))
}

/// The slice of `output` holding the final answer, plus whether an explicit
/// marker was found. Prefers the last `answer:` line, then the last `\boxed{}`,
/// then the last `####` line; otherwise the last non-empty line (no marker).
/// Lowercased — case is irrelevant to extraction. Marker regions are bounded to
/// their line so trailing reasoning can't hijack the number.
fn answer_region(output: &str) -> (String, bool) {
    let lower = output.to_lowercase();
    if let Some(pos) = lower.rfind("answer") {
        let rest = lower.get(pos..).unwrap_or("");
        let line = rest.split('\n').next().unwrap_or(rest);
        return (line.to_owned(), true);
    }
    if let Some(pos) = lower.rfind("\\boxed{") {
        let rest = lower.get(pos + "\\boxed{".len()..).unwrap_or("");
        if let Some(end) = rest.find('}') {
            return (rest.get(..end).unwrap_or("").to_owned(), true);
        }
    }
    if let Some(pos) = lower.rfind("####") {
        let rest = lower.get(pos + 4..).unwrap_or("");
        let line = rest.split('\n').next().unwrap_or(rest);
        return (line.to_owned(), true);
    }
    let last = lower
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("");
    (last.to_owned(), false)
}

/// Parse a whole string as a number, tolerating `$`, `,`, `%`, and surrounding
/// spaces. Returns `None` for anything non-numeric (e.g. "Dan", "10:25").
fn parse_number(s: &str) -> Option<f64> {
    let cleaned: String = s
        .chars()
        .filter(|c| !matches!(c, '$' | ',' | '%' | ' ' | '\t'))
        .collect();
    cleaned.parse::<f64>().ok()
}

/// All numbers appearing in `s` (commas stripped), left to right.
fn numbers_in(s: &str) -> Vec<f64> {
    // Drop commas so thousands separators ("1,070") don't split into two numbers,
    // while a comma list ("48, 24") keeps its space and stays two numbers.
    let cleaned: String = s.chars().filter(|&c| c != ',').collect();
    cleaned
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .filter_map(|tok| {
            let trimmed = tok.trim_matches(|c| c == '.' || c == '-');
            if trimmed.is_empty() {
                None
            } else {
                tok.parse::<f64>().ok()
            }
        })
        .collect()
}

fn last_number(s: &str) -> Option<f64> {
    numbers_in(s).into_iter().next_back()
}

/// Lowercase alphanumerics + single spaces; everything else dropped.
fn normalize_word(s: &str) -> String {
    let mapped: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect();
    mapped.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn problem_set_is_diverse_and_sized() {
        let problems = problem_set();
        assert_eq!(problems.len(), 50, "expected 50 problems");
        let categories: std::collections::HashSet<_> =
            problems.iter().map(|p| p.category).collect();
        assert!(
            categories.len() >= 8,
            "expected >= 8 categories, got {}",
            categories.len()
        );
        // Every gold answer is non-empty.
        assert!(problems.iter().all(|p| !p.answer.is_empty()));
    }

    #[test]
    fn grade_numeric_formats() {
        // The same gold (72) accepted across the answer formats models emit.
        assert!(grade("...so the total is 72.\nAnswer: 72", "72"));
        assert!(grade("The answer is 72.", "72"));
        assert!(grade("Total = 48 + 24 = 72\n\\boxed{72}", "72"));
        assert!(grade("reasoning...\n#### 72", "72"));
        assert!(grade("Answer: 72 clips", "72"));
        // Trailing prose / units / currency / commas.
        assert!(grade("Answer: $93", "93"));
        assert!(grade("Answer: 1,070 books", "1070"));
        assert!(grade("Answer: 4.80", "4.8"));
        // Wrong answers fail.
        assert!(!grade("Answer: 96", "72"));
        assert!(!grade("Answer: 48 + 48 = 96", "72"));
        // No explicit marker: fall back to the last number in the trace.
        assert!(grade("...adding gives 48 + 24 = 72.", "72"));
    }

    #[test]
    fn grade_word_answers() {
        assert!(grade("Answer: Dan", "Dan"));
        assert!(grade("So the shortest person is dan.", "Dan"));
        assert!(grade("Answer: uncle", "uncle"));
        assert!(grade("Answer: Saturday", "Saturday"));
        assert!(grade("Answer: yes", "yes"));
        assert!(!grade("Answer: no", "yes"));
        assert!(!grade("Answer: Beth", "Dan"));
    }

    #[test]
    fn extract_number_carries_state() {
        assert_eq!(extract_number("running total ... Answer: 53"), Some(53.0));
        assert_eq!(extract_number("50 - 5 = 45, then 45 + 4 = 49."), Some(49.0));
        assert_eq!(extract_number("no numbers here"), None);
    }

    #[test]
    fn grade_picks_answer_region_over_scratch() {
        // Lots of distractor numbers earlier; the Answer: line decides.
        let out = "Step 1: 48. Step 2: 24. Some 100 and 8 here.\nAnswer: 72";
        assert!(grade(out, "72"));
        assert!(!grade(out, "48"));
    }
}
