// Chapter 4: Operators
// Arithmetic, logical, and bitwise operators

const std = @import("std");

pub fn main() void {
    arithmetic_operators();
    comparison_operators();
    logical_operators();
    bitwise_operators();
}

fn arithmetic_operators() void {
    std.debug.print("=== Arithmetic Operators ===\n", .{});

    const a: i32 = 10;
    const b: i32 = 3;

    // Basic arithmetic
    std.debug.print("{d} + {d} = {d}\n", .{ a, b, a + b }); // Addition
    std.debug.print("{d} - {d} = {d}\n", .{ a, b, a - b }); // Subtraction
    std.debug.print("{d} * {d} = {d}\n", .{ a, b, a * b }); // Multiplication
    std.debug.print("{d} / {d} = {d}\n", .{ a, b, @divTrunc(a, b) }); // Division
    std.debug.print("{d} % {d} = {d}\n", .{ a, b, @mod(a, b) }); // Modulo

    // Negation
    std.debug.print("-{d} = {d}\n", .{ a, -a });

    // Wrapping arithmetic (for when overflow is intentional)
    var wrap: u8 = 255;
    wrap +%= 1; // Wraps to 0
    std.debug.print("255 +% 1 = {d}\n", .{wrap});

    // Saturating arithmetic (clamps at min/max)
    var sat: u8 = 250;
    sat +|= 10; // Saturates at 255
    std.debug.print("250 +| 10 = {d}\n", .{sat});
}

fn comparison_operators() void {
    std.debug.print("\n=== Comparison Operators ===\n", .{});

    const x: i32 = 5;
    const y: i32 = 10;

    std.debug.print("{d} == {d}: {}\n", .{ x, y, x == y }); // Equal
    std.debug.print("{d} != {d}: {}\n", .{ x, y, x != y }); // Not equal
    std.debug.print("{d} < {d}: {}\n", .{ x, y, x < y }); // Less than
    std.debug.print("{d} > {d}: {}\n", .{ x, y, x > y }); // Greater than
    std.debug.print("{d} <= {d}: {}\n", .{ x, y, x <= y }); // Less or equal
    std.debug.print("{d} >= {d}: {}\n", .{ x, y, x >= y }); // Greater or equal
}

fn logical_operators() void {
    std.debug.print("\n=== Logical Operators ===\n", .{});

    const t = true;
    const f = false;

    std.debug.print("{} and {}: {}\n", .{ t, f, t and f }); // Logical AND
    std.debug.print("{} or {}: {}\n", .{ t, f, t or f }); // Logical OR
    std.debug.print("not {}: {}\n", .{ t, !t }); // Logical NOT

    // Short-circuit evaluation
    const result = f and unreachable_func(); // unreachable_func never called
    std.debug.print("Short-circuit result: {}\n", .{result});
}

fn unreachable_func() bool {
    @panic("This should never be called!");
}

fn bitwise_operators() void {
    std.debug.print("\n=== Bitwise Operators ===\n", .{});

    const a: u8 = 0b11001010;
    const b: u8 = 0b10101100;

    std.debug.print("a = 0b{b:0>8}\n", .{a});
    std.debug.print("b = 0b{b:0>8}\n", .{b});

    std.debug.print("a & b  = 0b{b:0>8}\n", .{a & b}); // AND
    std.debug.print("a | b  = 0b{b:0>8}\n", .{a | b}); // OR
    std.debug.print("a ^ b  = 0b{b:0>8}\n", .{a ^ b}); // XOR
    std.debug.print("~a     = 0b{b:0>8}\n", .{~a}); // NOT

    // Bit shifts
    const c: u8 = 0b00001111;
    std.debug.print("c << 2 = 0b{b:0>8}\n", .{c << 2}); // Left shift
    std.debug.print("c >> 2 = 0b{b:0>8}\n", .{c >> 2}); // Right shift
}
