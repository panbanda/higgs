/**
 * Built-in tools the desktop app executes locally so tool-call round trips
 * are visible end to end without any external service.
 */

export interface ToolDefinition {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export const TOOL_DEFINITIONS: ToolDefinition[] = [
  {
    type: "function",
    function: {
      name: "get_current_time",
      description: "Get the current local date and time, optionally in a specific IANA timezone.",
      parameters: {
        type: "object",
        properties: {
          timezone: {
            type: "string",
            description: "IANA timezone such as 'America/Los_Angeles'. Defaults to the local timezone.",
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "calculate",
      description:
        "Evaluate an arithmetic expression. Supports + - * / % ^, parentheses, and the functions sqrt, abs, round, floor, ceil, sin, cos, tan, log, ln, exp, min, max.",
      parameters: {
        type: "object",
        properties: {
          expression: { type: "string", description: "The expression to evaluate, e.g. '(3 + 4) * sqrt(2)'." },
        },
        required: ["expression"],
      },
    },
  },
];

export async function executeTool(name: string, rawArguments: string): Promise<string> {
  let args: Record<string, unknown> = {};
  if (rawArguments.trim()) {
    try {
      args = JSON.parse(rawArguments) as Record<string, unknown>;
    } catch (error) {
      throw new Error(`Invalid JSON arguments: ${(error as Error).message}`);
    }
  }
  switch (name) {
    case "get_current_time":
      return currentTime(typeof args.timezone === "string" ? args.timezone : undefined);
    case "calculate": {
      if (typeof args.expression !== "string") throw new Error("Missing 'expression' argument");
      return JSON.stringify({ expression: args.expression, result: evaluate(args.expression) });
    }
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

function currentTime(timezone?: string): string {
  const now = new Date();
  const formatter = new Intl.DateTimeFormat("en-US", {
    dateStyle: "full",
    timeStyle: "long",
    timeZone: timezone,
  });
  return JSON.stringify({
    iso: now.toISOString(),
    formatted: formatter.format(now),
    timezone: timezone ?? Intl.DateTimeFormat().resolvedOptions().timeZone,
  });
}

type Token = { kind: "num"; value: number } | { kind: "id"; value: string } | { kind: "op"; value: string };

function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let index = 0;
  while (index < source.length) {
    const char = source[index];
    if (/\s/.test(char)) {
      index += 1;
    } else if (/[0-9.]/.test(char)) {
      const match = /^[0-9]*\.?[0-9]+(e[+-]?[0-9]+)?|^[0-9]+\.?/i.exec(source.slice(index));
      if (!match) throw new Error(`Bad number at position ${index}`);
      tokens.push({ kind: "num", value: Number(match[0]) });
      index += match[0].length;
    } else if (/[a-z_]/i.test(char)) {
      const match = /^[a-z_][a-z0-9_]*/i.exec(source.slice(index));
      if (!match) throw new Error(`Bad identifier at position ${index}`);
      tokens.push({ kind: "id", value: match[0].toLowerCase() });
      index += match[0].length;
    } else if ("+-*/%^(),".includes(char)) {
      tokens.push({ kind: "op", value: char });
      index += 1;
    } else {
      throw new Error(`Unexpected character '${char}' at position ${index}`);
    }
  }
  return tokens;
}

// Null-prototype records so a model-supplied identifier like "constructor" or
// "toString" can't resolve to an inherited Object.prototype member.
const CONSTANTS: Record<string, number> = Object.assign(Object.create(null), { pi: Math.PI, e: Math.E });
const FUNCTIONS: Record<string, (...args: number[]) => number> = Object.assign(Object.create(null), {
  sqrt: Math.sqrt,
  abs: Math.abs,
  round: Math.round,
  floor: Math.floor,
  ceil: Math.ceil,
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  log: Math.log10,
  ln: Math.log,
  exp: Math.exp,
  min: Math.min,
  max: Math.max,
});

/** Recursive-descent evaluator; no `eval`, so model-supplied input stays inert. */
export function evaluate(expression: string): number {
  const tokens = tokenize(expression);
  let position = 0;
  const peek = () => tokens[position];
  const take = () => tokens[position++];
  const expectOp = (value: string) => {
    const token = take();
    if (!token || token.kind !== "op" || token.value !== value) throw new Error(`Expected '${value}'`);
  };

  function parseExpression(): number {
    let left = parseTerm();
    for (;;) {
      const token = peek();
      if (token?.kind === "op" && (token.value === "+" || token.value === "-")) {
        take();
        const right = parseTerm();
        left = token.value === "+" ? left + right : left - right;
      } else {
        return left;
      }
    }
  }

  function parseTerm(): number {
    let left = parseUnary();
    for (;;) {
      const token = peek();
      if (token?.kind === "op" && "*/%".includes(token.value)) {
        take();
        const right = parseUnary();
        left = token.value === "*" ? left * right : token.value === "/" ? left / right : left % right;
      } else {
        return left;
      }
    }
  }

  function parseUnary(): number {
    const token = peek();
    if (token?.kind === "op" && token.value === "-") {
      take();
      return -parseUnary();
    }
    if (token?.kind === "op" && token.value === "+") {
      take();
      return parseUnary();
    }
    return parsePower();
  }

  function parsePower(): number {
    const base = parsePrimary();
    const token = peek();
    if (token?.kind === "op" && token.value === "^") {
      take();
      return base ** parseUnary();
    }
    return base;
  }

  function parsePrimary(): number {
    const token = take();
    if (!token) throw new Error("Unexpected end of expression");
    if (token.kind === "num") return token.value;
    if (token.kind === "op" && token.value === "(") {
      const value = parseExpression();
      expectOp(")");
      return value;
    }
    if (token.kind === "id") {
      const next = peek();
      if (next?.kind === "op" && next.value === "(") {
        take();
        const fn = FUNCTIONS[token.value];
        if (!fn) throw new Error(`Unknown function '${token.value}'`);
        const args: number[] = [];
        if (!(peek()?.kind === "op" && peek()?.value === ")")) {
          args.push(parseExpression());
          while (peek()?.kind === "op" && peek()?.value === ",") {
            take();
            args.push(parseExpression());
          }
        }
        expectOp(")");
        return fn(...args);
      }
      if (token.value in CONSTANTS) return CONSTANTS[token.value];
      throw new Error(`Unknown identifier '${token.value}'`);
    }
    throw new Error(`Unexpected token '${token.value}'`);
  }

  const result = parseExpression();
  if (position < tokens.length) throw new Error(`Unexpected trailing input '${tokens[position].value}'`);
  if (!Number.isFinite(result)) throw new Error("Result is not a finite number");
  return result;
}
