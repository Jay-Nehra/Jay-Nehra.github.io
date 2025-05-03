interface SectionHeaderProps {
  title: string;
  subtitle: string;
}

export function SectionHeader({ title, subtitle }: SectionHeaderProps) {
  return (
    <div className="max-w-3xl mx-auto text-center mb-16">
      <h2 className="font-poppins font-bold text-3xl sm:text-4xl mb-4">{title}</h2>
      <div className="w-20 h-1 bg-primary mx-auto mb-6"></div>
      <p className="text-lg text-gray-300 dark:text-gray-700">{subtitle}</p>
    </div>
  );
}
