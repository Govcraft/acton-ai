function LogomarkPaths() {
  return (
    <g fill="none" strokeLinejoin="round">
      {/* Three actor nodes forming a triangle */}
      <circle cx="18" cy="6" r="4" fill="#8B5CF6" />
      <circle cx="8" cy="28" r="4" fill="#8B5CF6" />
      <circle cx="28" cy="28" r="4" fill="#8B5CF6" />

      {/* Message arrows between actors */}
      {/* Top to bottom-left */}
      <path
        d="M15 9 L11 24"
        stroke="#8B5CF6"
        strokeWidth="2"
        markerEnd="url(#arrowhead)"
      />
      {/* Top to bottom-right */}
      <path
        d="M21 9 L25 24"
        stroke="#8B5CF6"
        strokeWidth="2"
        markerEnd="url(#arrowhead)"
      />
      {/* Bottom-left to bottom-right */}
      <path
        d="M12 28 L24 28"
        stroke="#8B5CF6"
        strokeWidth="2"
        markerEnd="url(#arrowhead)"
      />

      {/* Arrow marker definition */}
      <defs>
        <marker
          id="arrowhead"
          markerWidth="6"
          markerHeight="6"
          refX="5"
          refY="3"
          orient="auto"
        >
          <polygon points="0 0, 6 3, 0 6" fill="#8B5CF6" />
        </marker>
      </defs>
    </g>
  )
}

export function Logomark(props: React.ComponentPropsWithoutRef<'svg'>) {
  return (
    <svg aria-hidden="true" viewBox="0 0 36 36" fill="none" {...props}>
      <LogomarkPaths />
    </svg>
  )
}

export function Logo(props: React.ComponentPropsWithoutRef<'svg'>) {
  return (
    <svg aria-hidden="true" viewBox="0 0 140 36" fill="none" {...props}>
      <LogomarkPaths />
      {/* "acton-ai" wordmark */}
      <text
        x="42"
        y="24"
        fontFamily="var(--font-lexend), system-ui, sans-serif"
        fontSize="18"
        fontWeight="600"
        fill="currentColor"
      >
        acton-ai
      </text>
    </svg>
  )
}
