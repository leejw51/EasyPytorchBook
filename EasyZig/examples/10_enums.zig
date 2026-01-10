// Chapter 10: Enums
// Enumerated types in Zig

const std = @import("std");

// Basic enum
const Color = enum {
    red,
    green,
    blue,
    yellow,
    purple,
};

// Enum with explicit values
const HttpStatus = enum(u16) {
    ok = 200,
    created = 201,
    not_found = 404,
    internal_error = 500,
};

// Enum with methods
const Direction = enum {
    north,
    south,
    east,
    west,

    pub fn opposite(self: Direction) Direction {
        return switch (self) {
            .north => .south,
            .south => .north,
            .east => .west,
            .west => .east,
        };
    }

    pub fn toVector(self: Direction) struct { x: i32, y: i32 } {
        return switch (self) {
            .north => .{ .x = 0, .y = -1 },
            .south => .{ .x = 0, .y = 1 },
            .east => .{ .x = 1, .y = 0 },
            .west => .{ .x = -1, .y = 0 },
        };
    }
};

pub fn main() void {
    basic_enums();
    enum_values();
    enum_methods();
    enum_iteration();
}

fn basic_enums() void {
    std.debug.print("=== Basic Enums ===\n", .{});

    // Declare enum variable
    const favorite: Color = .blue;
    var current: Color = .red;

    std.debug.print("Favorite color: {}\n", .{favorite});
    std.debug.print("Current color: {}\n", .{current});

    // Change value
    current = .green;
    std.debug.print("Changed to: {}\n", .{current});

    // Compare enums
    if (favorite == .blue) {
        std.debug.print("Blue is the best!\n", .{});
    }

    // Switch on enum
    const message = switch (current) {
        .red => "Stop!",
        .green => "Go!",
        .blue => "Cool",
        .yellow => "Caution",
        .purple => "Royal",
    };
    std.debug.print("Message: {s}\n", .{message});
}

fn enum_values() void {
    std.debug.print("\n=== Enum Values ===\n", .{});

    // Get integer value
    const status: HttpStatus = .ok;
    const code = @intFromEnum(status);
    std.debug.print("HTTP {}: {}\n", .{ code, status });

    const not_found: HttpStatus = .not_found;
    std.debug.print("HTTP {}: {}\n", .{ @intFromEnum(not_found), not_found });

    // Convert integer to enum
    const from_int: HttpStatus = @enumFromInt(201);
    std.debug.print("From 201: {}\n", .{from_int});
}

fn enum_methods() void {
    std.debug.print("\n=== Enum Methods ===\n", .{});

    const dir: Direction = .north;
    std.debug.print("Direction: {}\n", .{dir});
    std.debug.print("Opposite: {}\n", .{dir.opposite()});

    const vec = dir.toVector();
    std.debug.print("Vector: ({d}, {d})\n", .{ vec.x, vec.y });

    // Chain method calls
    const south_vec = Direction.south.toVector();
    std.debug.print("South vector: ({d}, {d})\n", .{ south_vec.x, south_vec.y });
}

fn enum_iteration() void {
    std.debug.print("\n=== Enum Iteration ===\n", .{});

    // Iterate over all enum values
    std.debug.print("All colors: ", .{});
    inline for (std.meta.fields(Color)) |field| {
        std.debug.print("{s} ", .{field.name});
    }
    std.debug.print("\n", .{});

    // Count enum values
    const color_count = std.meta.fields(Color).len;
    std.debug.print("Number of colors: {d}\n", .{color_count});
}

// Non-exhaustive enum (can have unknown values)
const FileType = enum(u8) {
    text = 1,
    binary = 2,
    image = 3,
    _, // Non-exhaustive marker

    pub fn describe(self: FileType) []const u8 {
        return switch (self) {
            .text => "Text file",
            .binary => "Binary file",
            .image => "Image file",
            _ => "Unknown file type",
        };
    }
};

pub fn non_exhaustive_enum() void {
    std.debug.print("\n=== Non-Exhaustive Enum ===\n", .{});

    const known: FileType = .text;
    std.debug.print("{}: {s}\n", .{ known, known.describe() });

    // Create unknown value
    const unknown: FileType = @enumFromInt(99);
    std.debug.print("Unknown value: {s}\n", .{unknown.describe()});
}

// Enum as array index
const Weekday = enum(u3) {
    monday = 0,
    tuesday = 1,
    wednesday = 2,
    thursday = 3,
    friday = 4,
    saturday = 5,
    sunday = 6,
};

const hours_worked = [_]u8{ 8, 8, 8, 8, 8, 0, 0 };

pub fn enum_as_index() void {
    std.debug.print("\n=== Enum as Index ===\n", .{});

    const day: Weekday = .wednesday;
    const hours = hours_worked[@intFromEnum(day)];
    std.debug.print("Hours on {}: {d}\n", .{ day, hours });
}
