// Chapter 12: Control Flow
// Conditionals and loops in Zig

const std = @import("std");

pub fn main() void {
    if_expressions();
    switch_statements();
    while_loops();
    for_loops();
    labeled_blocks();
}

fn if_expressions() void {
    std.debug.print("=== If Expressions ===\n", .{});

    const x: i32 = 42;

    // Basic if
    if (x > 0) {
        std.debug.print("x is positive\n", .{});
    }

    // If-else
    if (x % 2 == 0) {
        std.debug.print("x is even\n", .{});
    } else {
        std.debug.print("x is odd\n", .{});
    }

    // If-else chain
    const score: u32 = 85;
    if (score >= 90) {
        std.debug.print("Grade: A\n", .{});
    } else if (score >= 80) {
        std.debug.print("Grade: B\n", .{});
    } else if (score >= 70) {
        std.debug.print("Grade: C\n", .{});
    } else {
        std.debug.print("Grade: F\n", .{});
    }

    // If as expression
    const abs_x = if (x >= 0) x else -x;
    std.debug.print("Absolute value: {d}\n", .{abs_x});

    // Ternary-like
    const message = if (x > 0) "positive" else "non-positive";
    std.debug.print("x is {s}\n", .{message});
}

fn switch_statements() void {
    std.debug.print("\n=== Switch Statements ===\n", .{});

    // Basic switch
    const day: u8 = 3;
    const day_name = switch (day) {
        1 => "Monday",
        2 => "Tuesday",
        3 => "Wednesday",
        4 => "Thursday",
        5 => "Friday",
        6, 7 => "Weekend", // Multiple values
        else => "Invalid",
    };
    std.debug.print("Day {d} is {s}\n", .{ day, day_name });

    // Range in switch
    const age: u32 = 25;
    const category = switch (age) {
        0...12 => "Child",
        13...19 => "Teenager",
        20...64 => "Adult",
        else => "Senior",
    };
    std.debug.print("Age {d}: {s}\n", .{ age, category });

    // Switch with capture
    const value: i32 = -5;
    switch (value) {
        1...100 => |v| std.debug.print("Positive value: {d}\n", .{v}),
        else => |v| std.debug.print("Other value: {d}\n", .{v}),
    }

    // Exhaustive switch (no else needed for enums)
    const Color = enum { red, green, blue };
    const color: Color = .green;
    switch (color) {
        .red => std.debug.print("Stop!\n", .{}),
        .green => std.debug.print("Go!\n", .{}),
        .blue => std.debug.print("Water\n", .{}),
    }
}

fn while_loops() void {
    std.debug.print("\n=== While Loops ===\n", .{});

    // Basic while
    var i: u32 = 0;
    std.debug.print("Count: ", .{});
    while (i < 5) {
        std.debug.print("{d} ", .{i});
        i += 1;
    }
    std.debug.print("\n", .{});

    // While with continue expression
    var j: u32 = 0;
    std.debug.print("Evens: ", .{});
    while (j < 10) : (j += 1) {
        if (j % 2 != 0) continue;
        std.debug.print("{d} ", .{j});
    }
    std.debug.print("\n", .{});

    // While with else
    var k: u32 = 0;
    const found = while (k < 5) : (k += 1) {
        if (k == 3) break true;
    } else false;
    std.debug.print("Found 3: {}\n", .{found});

    // Break with value
    var n: u32 = 0;
    const result = while (n < 100) : (n += 1) {
        if (n * n > 50) break n;
    } else 0;
    std.debug.print("First n where n² > 50: {d}\n", .{result});
}

fn for_loops() void {
    std.debug.print("\n=== For Loops ===\n", .{});

    // Iterate over array
    const numbers = [_]i32{ 10, 20, 30, 40, 50 };
    std.debug.print("Numbers: ", .{});
    for (numbers) |n| {
        std.debug.print("{d} ", .{n});
    }
    std.debug.print("\n", .{});

    // With index
    std.debug.print("Indexed: ", .{});
    for (numbers, 0..) |n, i| {
        std.debug.print("[{d}]={d} ", .{ i, n });
    }
    std.debug.print("\n", .{});

    // Iterate over range
    std.debug.print("Range 0..5: ", .{});
    for (0..5) |i| {
        std.debug.print("{d} ", .{i});
    }
    std.debug.print("\n", .{});

    // Multiple arrays
    const a = [_]i32{ 1, 2, 3 };
    const b = [_]i32{ 10, 20, 30 };
    std.debug.print("Pairs: ", .{});
    for (a, b) |x, y| {
        std.debug.print("({d},{d}) ", .{ x, y });
    }
    std.debug.print("\n", .{});

    // Modify elements (pointer capture)
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    for (&arr) |*item| {
        item.* *= 2;
    }
    std.debug.print("Doubled: {any}\n", .{arr});
}

fn labeled_blocks() void {
    std.debug.print("\n=== Labeled Blocks ===\n", .{});

    // Labeled block as expression
    const result = blk: {
        var sum: i32 = 0;
        for (0..10) |i| {
            sum += @as(i32, @intCast(i));
        }
        break :blk sum;
    };
    std.debug.print("Sum 0-9: {d}\n", .{result});

    // Nested loop with labels
    std.debug.print("Matrix search:\n", .{});
    outer: for (0..3) |i| {
        for (0..3) |j| {
            if (i == 1 and j == 1) {
                std.debug.print("Found at ({d}, {d})\n", .{ i, j });
                break :outer;
            }
            std.debug.print("({d}, {d}) ", .{ i, j });
        }
        std.debug.print("\n", .{});
    }
}

// Defer and errdefer
pub fn defer_examples() !void {
    std.debug.print("\n=== Defer ===\n", .{});

    // Defer runs at scope exit
    {
        defer std.debug.print("3: Last\n", .{});
        defer std.debug.print("2: Second\n", .{});
        std.debug.print("1: First\n", .{});
    }

    // LIFO order: prints 1, 2, 3
}
