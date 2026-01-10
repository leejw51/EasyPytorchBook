// Chapter 11: Unions
// Tagged unions in Zig

const std = @import("std");

// Tagged union (discriminated union)
const Number = union(enum) {
    int: i64,
    float: f64,
    invalid: void,

    pub fn add(self: Number, other: Number) Number {
        return switch (self) {
            .int => |a| switch (other) {
                .int => |b| .{ .int = a + b },
                .float => |b| .{ .float = @as(f64, @floatFromInt(a)) + b },
                .invalid => .invalid,
            },
            .float => |a| switch (other) {
                .int => |b| .{ .float = a + @as(f64, @floatFromInt(b)) },
                .float => |b| .{ .float = a + b },
                .invalid => .invalid,
            },
            .invalid => .invalid,
        };
    }

    pub fn print(self: Number) void {
        switch (self) {
            .int => |v| std.debug.print("int: {d}\n", .{v}),
            .float => |v| std.debug.print("float: {d}\n", .{v}),
            .invalid => std.debug.print("invalid\n", .{}),
        }
    }
};

// Untagged union (bare union)
const Converter = extern union {
    int_val: i32,
    bytes: [4]u8,
};

// Complex tagged union for AST-like structure
const Expr = union(enum) {
    literal: i64,
    add: *const BinaryOp,
    multiply: *const BinaryOp,
    negate: *const Expr,
};

const BinaryOp = struct {
    left: Expr,
    right: Expr,
};

pub fn main() void {
    basic_unions();
    union_switch();
    union_methods();
    bare_unions();
}

fn basic_unions() void {
    std.debug.print("=== Basic Unions ===\n", .{});

    // Create union values
    const a: Number = .{ .int = 42 };
    const b: Number = .{ .float = 3.14 };
    const c: Number = .invalid;

    std.debug.print("a = ", .{});
    a.print();

    std.debug.print("b = ", .{});
    b.print();

    std.debug.print("c = ", .{});
    c.print();

    // Check active tag
    std.debug.print("a is int: {}\n", .{a == .int});
    std.debug.print("b is int: {}\n", .{b == .int});
}

fn union_switch() void {
    std.debug.print("\n=== Union Switch ===\n", .{});

    const values = [_]Number{
        .{ .int = 10 },
        .{ .float = 2.5 },
        .invalid,
        .{ .int = -5 },
    };

    for (values) |value| {
        switch (value) {
            .int => |n| std.debug.print("Integer: {d}\n", .{n}),
            .float => |f| std.debug.print("Float: {d:.2}\n", .{f}),
            .invalid => std.debug.print("Invalid value\n", .{}),
        }
    }
}

fn union_methods() void {
    std.debug.print("\n=== Union Methods ===\n", .{});

    const a: Number = .{ .int = 10 };
    const b: Number = .{ .int = 20 };
    const c: Number = .{ .float = 5.5 };

    std.debug.print("a + b = ", .{});
    a.add(b).print();

    std.debug.print("a + c = ", .{});
    a.add(c).print();

    std.debug.print("c + c = ", .{});
    c.add(c).print();
}

fn bare_unions() void {
    std.debug.print("\n=== Bare Unions ===\n", .{});

    // Type punning with bare union
    var conv = Converter{ .int_val = 0x41424344 };
    std.debug.print("Integer: 0x{x}\n", .{conv.int_val});
    std.debug.print("Bytes: {c}{c}{c}{c}\n", .{
        conv.bytes[0],
        conv.bytes[1],
        conv.bytes[2],
        conv.bytes[3],
    });

    // Modify through different view
    conv.bytes[0] = 'Z';
    std.debug.print("Modified integer: 0x{x}\n", .{conv.int_val});
}

// Optional-like pattern with union
const Maybe = union(enum) {
    value: i32,
    nothing: void,

    pub fn unwrap(self: Maybe) ?i32 {
        return switch (self) {
            .value => |v| v,
            .nothing => null,
        };
    }

    pub fn map(self: Maybe, f: *const fn (i32) i32) Maybe {
        return switch (self) {
            .value => |v| .{ .value = f(v) },
            .nothing => .nothing,
        };
    }
};

fn double(x: i32) i32 {
    return x * 2;
}

pub fn optional_pattern() void {
    std.debug.print("\n=== Optional Pattern ===\n", .{});

    const some: Maybe = .{ .value = 42 };
    const none: Maybe = .nothing;

    std.debug.print("some.unwrap() = {?d}\n", .{some.unwrap()});
    std.debug.print("none.unwrap() = {?d}\n", .{none.unwrap()});

    const doubled = some.map(double);
    std.debug.print("doubled = {?d}\n", .{doubled.unwrap()});
}

// Result-like pattern
const Result = union(enum) {
    ok: i32,
    err: []const u8,

    pub fn isOk(self: Result) bool {
        return self == .ok;
    }
};

pub fn result_pattern() void {
    std.debug.print("\n=== Result Pattern ===\n", .{});

    const success: Result = .{ .ok = 42 };
    const failure: Result = .{ .err = "Something went wrong" };

    if (success.isOk()) {
        std.debug.print("Success: {d}\n", .{success.ok});
    }

    switch (failure) {
        .ok => |v| std.debug.print("Value: {d}\n", .{v}),
        .err => |e| std.debug.print("Error: {s}\n", .{e}),
    }
}
