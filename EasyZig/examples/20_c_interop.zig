// Chapter 20: C Interoperability
// Calling C from Zig and exporting to C

const std = @import("std");
const c = @cImport({
    @cInclude("stdio.h");
    @cInclude("stdlib.h");
    @cInclude("string.h");
    @cInclude("math.h");
});

pub fn main() void {
    c_types();
    c_functions();
    c_strings();
    c_memory();
}

fn c_types() void {
    std.debug.print("=== C Types ===\n", .{});

    // C integer types
    const a: c_int = 42;
    const b: c_long = 1000000;
    const d: c_uint = 255;

    std.debug.print("c_int: {d}\n", .{a});
    std.debug.print("c_long: {d}\n", .{b});
    std.debug.print("c_uint: {d}\n", .{d});

    // Size of C types
    std.debug.print("\nC type sizes:\n", .{});
    std.debug.print("  c_int: {d} bytes\n", .{@sizeOf(c_int)});
    std.debug.print("  c_long: {d} bytes\n", .{@sizeOf(c_long)});
    std.debug.print("  c_longlong: {d} bytes\n", .{@sizeOf(c_longlong)});

    // C floating point
    const f: c.double = 3.14159;
    std.debug.print("c.double: {d}\n", .{f});
}

fn c_functions() void {
    std.debug.print("\n=== C Functions ===\n", .{});

    // Call C math functions
    const x: c.double = 2.0;

    const sqrt_x = c.sqrt(x);
    const pow_x = c.pow(x, 3.0);
    const sin_x = c.sin(x);
    const cos_x = c.cos(x);

    std.debug.print("sqrt({d}) = {d}\n", .{ x, sqrt_x });
    std.debug.print("pow({d}, 3) = {d}\n", .{ x, pow_x });
    std.debug.print("sin({d}) = {d}\n", .{ x, sin_x });
    std.debug.print("cos({d}) = {d}\n", .{ x, cos_x });

    // abs function
    const neg: c_int = -42;
    const abs_val = c.abs(neg);
    std.debug.print("abs({d}) = {d}\n", .{ neg, abs_val });
}

fn c_strings() void {
    std.debug.print("\n=== C Strings ===\n", .{});

    // Zig string to C string
    const zig_str: [:0]const u8 = "Hello from Zig!";
    const c_str: [*c]const u8 = zig_str.ptr;

    // Print using C printf
    _ = c.printf("C printf: %s\n", c_str);

    // String length using C strlen
    const len = c.strlen(c_str);
    std.debug.print("strlen: {d}\n", .{len});

    // String comparison
    const str1: [*c]const u8 = "apple";
    const str2: [*c]const u8 = "banana";
    const cmp = c.strcmp(str1, str2);
    std.debug.print("strcmp(apple, banana) = {d}\n", .{cmp});
}

fn c_memory() void {
    std.debug.print("\n=== C Memory ===\n", .{});

    // Allocate with malloc
    const size: usize = 10;
    const ptr: ?[*]u8 = @ptrCast(c.malloc(size));

    if (ptr) |p| {
        // Use memset to initialize
        _ = c.memset(p, 'A', size);

        std.debug.print("Allocated and initialized: ", .{});
        for (0..size) |i| {
            std.debug.print("{c}", .{p[i]});
        }
        std.debug.print("\n", .{});

        // Free memory
        c.free(p);
        std.debug.print("Memory freed\n", .{});
    }
}

// Export functions to C
export fn zig_add(a: c_int, b: c_int) c_int {
    return a + b;
}

export fn zig_multiply(a: c_int, b: c_int) c_int {
    return a * b;
}

export fn zig_factorial(n: c_uint) c_uint {
    if (n <= 1) return 1;
    return n * zig_factorial(n - 1);
}

// Extern struct compatible with C
const CPoint = extern struct {
    x: c_int,
    y: c_int,
};

export fn create_point(x: c_int, y: c_int) CPoint {
    return CPoint{ .x = x, .y = y };
}

export fn point_distance(p1: CPoint, p2: CPoint) c.double {
    const dx: c.double = @floatFromInt(p2.x - p1.x);
    const dy: c.double = @floatFromInt(p2.y - p1.y);
    return c.sqrt(dx * dx + dy * dy);
}

// Callback function type
const CCallback = *const fn (c_int) callconv(.C) c_int;

export fn apply_callback(value: c_int, callback: CCallback) c_int {
    return callback(value);
}

pub fn export_example() void {
    std.debug.print("\n=== Exported Functions ===\n", .{});

    // Test exported functions
    std.debug.print("zig_add(5, 3) = {d}\n", .{zig_add(5, 3)});
    std.debug.print("zig_multiply(4, 7) = {d}\n", .{zig_multiply(4, 7)});
    std.debug.print("zig_factorial(5) = {d}\n", .{zig_factorial(5)});

    // Test point functions
    const p1 = create_point(0, 0);
    const p2 = create_point(3, 4);
    const dist = point_distance(p1, p2);
    std.debug.print("Distance from (0,0) to (3,4) = {d}\n", .{dist});
}
