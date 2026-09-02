import type { ReactNode } from "react";

export interface Column<T> {
  key: string;
  header: ReactNode;
  render: (row: T) => ReactNode;
  align?: "left" | "right";
  width?: string;
}

interface Props<T> {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  empty?: ReactNode;
  onRowClick?: (row: T) => void;
  selectedKey?: string | null;
  dense?: boolean;
}

export function DataTable<T>({ columns, rows, rowKey, empty = "No data", onRowClick, selectedKey, dense }: Props<T>) {
  return (
    <div className="table-wrap">
      <table className={`data-table ${dense ? "dense" : ""}`}>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key} style={{ textAlign: column.align ?? "left", width: column.width }}>
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr>
              <td colSpan={columns.length} className="muted">
                {empty}
              </td>
            </tr>
          )}
          {rows.map((row) => {
            const key = rowKey(row);
            return (
              <tr key={key} className={`${onRowClick ? "clickable" : ""} ${selectedKey === key ? "selected" : ""}`} onClick={() => onRowClick?.(row)}>
                {columns.map((column) => (
                  <td key={column.key} style={{ textAlign: column.align ?? "left" }}>
                    {column.render(row)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
