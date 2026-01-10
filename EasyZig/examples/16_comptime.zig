// Chapter 16: Comptime
// Compile-time execution in Zig

const std = @import("std");

// Compile-time constants
const BUFFER_SIZE = 1024;
const KILOBYTE = 1024;
const MEGABYTE = KILOBYTE * 1024;

// Compile-time function execution
fn factorial(n: u64) u64 {
    if (n == 0) return 1;
    return n * factorial(n - 1);
}

// Pre-computed at compile time
const FACT_10 = factorial(10);
const FACT_5 = factorial(5);

pub fn main() void {
    comptime_basics();
    comptime_branching();
    comptime_loops();
    type_reflection();
}

fn comptime_basics() void {
    std.debug.print("=== Comptime Basics ===\n", .{});

    // Compile-time computed constants
    std.debug.print("1 KB = {d} bytes\n", .{KILOBYTE});
    std.debug.print("1 MB = {d} bytes\n", .{MEGABYTE});

    // Compile-time function results
    std.debug.print("10! = {d}\n", .{FACT_10});
    std.debug.print("5! = {d}\n", .{FACT_5});

    // Comptime block in local scope
    const sum_1_to_10 = comptime blk: {
        var sum: u32 = 0;
        for (1..11) |i| {
            sum += @as(u32, @intCast(i));
        }
        break :blk sum;
    };
    std.debug.print("Sum 1-10 (comptime): {d}\n", .{sum_1_to_10});

    // Comptime array generation
    const squares = comptime blk: {
        var arr: [10]i32 = undefined;
        for (0..10) |i| {
            arr[i] = @as(i32, @intCast(i * i));
        }
        break :blk arr;
    };
    std.debug.print("Squares: {any}\n", .{squares});
}

fn comptime_branching() void {
    std.debug.print("\n=== Comptime Branching ===\n", .{});

    // Conditional compilation
    const ptr_size = @sizeOf(*u8);
    const arch_name = if (ptr_size == 8) "64-bit" else "32-bit";
    std.debug.print("Architecture: {s}\n", .{arch_name});

    // Build mode detection
    const is_debug = @import("builtin").mode == .Debug;
    if (is_debug) {
        std.debug.print("Running in debug mode\n", .{});
    } else {
        std.debug.print("Running in release mode\n", .{});
    }
}

fn comptime_loops() void {
    std.debug.print("\n=== Comptime Loops ===\n", .{});

    // Inline for (unrolled at compile time)
    const tuple = .{ "hello", 42, true, 3.14 };

    std.debug.print("Tuple contents:\n", .{});
    inline for (tuple, 0..) |item, i| {
        std.debug.print("  [{d}] = {any}\n", .{ i, item });
    }

    // Generate code for multiple types
    const types = .{ u8, u16, u32, u64 };
    std.debug.print("Type sizes: ", .{});
    inline for (types) |T| {
        std.debug.print("{s}={d} ", .{ @typeName(T), @sizeOf(T) });
    }
    std.debug.print("\n", .{});
}

fn type_reflection() void {
    std.debug.print("\n=== Type Reflection ===\n", .{});

    // Get type information
    const T = i32;
    std.debug.print("Type: {s}\n", .{@typeName(T)});
    std.debug.print("Size: {d} bytes\n", .{@sizeOf(T)});
    std.debug.print("Alignment: {d}\n", .{@alignOf(T)});

    // Struct field introspection
    const Point = struct { x: f32, y: f32, z: f32 };

    std.debug.print("\nPoint fields:\n", .{});
    inline for (std.meta.fields(Point)) |field| {
        std.debug.print("  {s}: {s}\n", .{ field.name, @typeName(field.type) });
    }

    // Create instance using field info
    var point: Point = undefined;
    inline for (std.meta.fields(Point), 0..) |field, i| {
        @field(point, field.name) = @as(f32, @floatFromInt(i + 1));
    }
    std.debug.print("Point: ({d}, {d}, {d})\n", .{ point.x, point.y, point.z });
}

// Comptime string operations
fn comptimeConcat(comptime a: []const u8, comptime b: []const u8) *const [a.len + b.len]u8 {
    return a ++ b;
}

// Comptime type selection
fn selectType(comptime use_float: bool) type {
    return if (use_float) f64 else i64;
}

pub fn advanced_comptime() void {
    std.debug.print("\n=== Advanced Comptime ===\n", .{});

    const greeting = comptimeConcat("Hello, ", "World!");
    std.debug.print("{s}\n", .{greeting});

    const FloatType = selectType(true);
    const IntType = selectType(false);

    const f: FloatType = 3.14;
    const i: IntType = 42;
    std.debug.print("Float: {d}, Int: {d}\n", .{ f, i });
}
